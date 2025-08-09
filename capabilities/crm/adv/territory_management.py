"""
APG Customer Relationship Management - Territory Management Module

Advanced territory management system for sales territory assignment, geographic
coverage analysis, and account distribution optimization.

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

from .models import CRMAccount, CRMContact
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class TerritoryType(str, Enum):
	"""Types of territories"""
	GEOGRAPHIC = "geographic"
	INDUSTRY = "industry"
	ACCOUNT_SIZE = "account_size"
	PRODUCT = "product"
	CHANNEL = "channel"
	HYBRID = "hybrid"


class TerritoryStatus(str, Enum):
	"""Territory status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	PLANNING = "planning"
	ARCHIVED = "archived"


class AssignmentType(str, Enum):
	"""Types of account assignments"""
	PRIMARY = "primary"
	SECONDARY = "secondary"
	OVERLAY = "overlay"
	SHARED = "shared"


class Territory(BaseModel):
	"""Model for sales territories"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	territory_name: str = Field(..., min_length=1, max_length=200, description="Territory name")
	territory_code: Optional[str] = Field(None, max_length=50, description="Territory code")
	territory_type: TerritoryType = Field(..., description="Type of territory")
	status: TerritoryStatus = Field(default=TerritoryStatus.ACTIVE)
	
	# Assignment
	owner_id: str = Field(..., description="Territory owner/manager")
	sales_rep_ids: List[str] = Field(default_factory=list, description="Assigned sales representatives")
	
	# Geographic criteria
	countries: List[str] = Field(default_factory=list, description="Country codes")
	states_provinces: List[str] = Field(default_factory=list, description="State/province codes")
	cities: List[str] = Field(default_factory=list, description="City names")
	postal_codes: List[str] = Field(default_factory=list, description="Postal/ZIP codes")
	
	# Business criteria
	industries: List[str] = Field(default_factory=list, description="Industry classifications")
	company_size_min: Optional[int] = Field(None, description="Minimum company size (employees)")
	company_size_max: Optional[int] = Field(None, description="Maximum company size (employees)")
	revenue_min: Optional[float] = Field(None, description="Minimum annual revenue")
	revenue_max: Optional[float] = Field(None, description="Maximum annual revenue")
	
	# Product/service criteria
	product_lines: List[str] = Field(default_factory=list, description="Product line focus")
	service_types: List[str] = Field(default_factory=list, description="Service type focus")
	
	# Goals and metrics
	annual_quota: Optional[float] = Field(None, description="Annual sales quota")
	account_target: Optional[int] = Field(None, description="Target number of accounts")
	
	# Metadata
	description: Optional[str] = Field(None, max_length=2000, description="Territory description")
	notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")
	rules: Dict[str, Any] = Field(default_factory=dict, description="Assignment rules")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	# Performance tracking
	current_accounts: int = Field(default=0, description="Current account count")
	current_revenue: float = Field(default=0.0, description="Current territory revenue")
	quota_achievement: float = Field(default=0.0, description="Quota achievement percentage")
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., description="Creator user ID")
	updated_by: str = Field(..., description="Last updater user ID")
	version: int = Field(default=1)


class AccountTerritoryAssignment(BaseModel):
	"""Model for account-territory assignments"""
	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant identifier")
	account_id: str = Field(..., description="Account identifier")
	territory_id: str = Field(..., description="Territory identifier")
	assignment_type: AssignmentType = Field(default=AssignmentType.PRIMARY)
	
	# Assignment details
	assigned_by: str = Field(..., description="User who made the assignment")
	assignment_reason: Optional[str] = Field(None, description="Reason for assignment")
	effective_date: datetime = Field(default_factory=datetime.utcnow)
	expiry_date: Optional[datetime] = Field(None, description="Assignment expiry date")
	
	# Performance tracking
	assignment_score: Optional[float] = Field(None, description="Assignment quality score")
	
	# Metadata
	notes: Optional[str] = Field(None, max_length=1000, description="Assignment notes")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	
	# Audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., description="Creator user ID")
	updated_by: str = Field(..., description="Last updater user ID")
	version: int = Field(default=1)


class TerritoryAnalytics(BaseModel):
	"""Analytics model for territory performance"""
	territory_id: str
	territory_name: str
	total_accounts: int = 0
	total_revenue: float = 0.0
	quota_achievement: float = 0.0
	account_distribution: Dict[str, int] = Field(default_factory=dict)
	revenue_distribution: Dict[str, float] = Field(default_factory=dict)
	performance_trends: List[Dict[str, Any]] = Field(default_factory=list)
	coverage_gaps: List[Dict[str, Any]] = Field(default_factory=list)


class TerritoryError(Exception):
	"""Base exception for territory operations"""
	pass


class TerritoryManager:
	"""
	Advanced territory management system for sales territory assignment,
	geographic coverage analysis, and account distribution optimization.
	"""
	
	def __init__(self, database_manager: DatabaseManager):
		"""
		Initialize territory manager
		
		Args:
			database_manager: Database manager instance
		"""
		self.db_manager = database_manager
		self.assignment_rules = self._load_assignment_rules()
	
	def _load_assignment_rules(self) -> Dict[str, Any]:
		"""Load territory assignment rules and algorithms"""
		return {
			"geographic_priority": ["postal_codes", "cities", "states_provinces", "countries"],
			"business_criteria_weights": {
				"industry_match": 0.4,
				"company_size_fit": 0.3,
				"revenue_potential": 0.3
			},
			"auto_assignment": {
				"enabled": True,
				"min_confidence": 0.7,
				"conflict_resolution": "primary_territory_wins"
			}
		}
	
	# ================================
	# Territory Management
	# ================================
	
	async def create_territory(
		self,
		territory_data: Dict[str, Any],
		tenant_id: str,
		created_by: str
	) -> Territory:
		"""
		Create a new territory
		
		Args:
			territory_data: Territory information
			tenant_id: Tenant identifier
			created_by: User creating the territory
			
		Returns:
			Created territory object
		"""
		try:
			logger.info(f"🗺️ Creating new territory")
			
			# Add audit fields
			territory_data.update({
				"tenant_id": tenant_id,
				"created_by": created_by,
				"updated_by": created_by
			})
			
			# Validate territory
			territory = Territory(**territory_data)
			
			# Check for duplicate territory name
			existing = await self._get_territory_by_name(territory.territory_name, tenant_id)
			if existing:
				raise TerritoryError(f"Territory with name '{territory.territory_name}' already exists")
			
			# Save to database
			saved_territory = await self._save_territory(territory)
			
			logger.info(f"✅ Created territory: {saved_territory.id}")
			return saved_territory
			
		except ValidationError as e:
			raise TerritoryError(f"Invalid territory data: {str(e)}")
		except Exception as e:
			logger.error(f"Failed to create territory: {str(e)}", exc_info=True)
			raise TerritoryError(f"Territory creation failed: {str(e)}")
	
	async def get_territory(
		self,
		territory_id: str,
		tenant_id: str
	) -> Optional[Territory]:
		"""
		Get territory by ID
		
		Args:
			territory_id: Territory identifier
			tenant_id: Tenant identifier
			
		Returns:
			Territory object or None
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_territories 
					WHERE id = $1 AND tenant_id = $2
				""", territory_id, tenant_id)
				
				if row:
					return self._row_to_territory(row)
				return None
				
		except Exception as e:
			logger.error(f"Failed to get territory: {str(e)}", exc_info=True)
			raise TerritoryError(f"Get territory failed: {str(e)}")
	
	async def list_territories(
		self,
		tenant_id: str,
		territory_type: Optional[TerritoryType] = None,
		status: Optional[TerritoryStatus] = None,
		owner_id: Optional[str] = None,
		include_metrics: bool = True
	) -> List[Dict[str, Any]]:
		"""
		List territories with optional filtering
		
		Args:
			tenant_id: Tenant identifier
			territory_type: Filter by territory type
			status: Filter by status
			owner_id: Filter by owner
			include_metrics: Include performance metrics
			
		Returns:
			List of territories with optional metrics
		"""
		try:
			logger.info(f"📋 Listing territories for tenant: {tenant_id}")
			
			async with self.db_manager.get_connection() as conn:
				# Build query
				query = "SELECT * FROM crm_territories WHERE tenant_id = $1"
				params = [tenant_id]
				param_counter = 2
				
				if territory_type:
					query += f" AND territory_type = ${param_counter}"
					params.append(territory_type.value)
					param_counter += 1
				
				if status:
					query += f" AND status = ${param_counter}"
					params.append(status.value)
					param_counter += 1
				
				if owner_id:
					query += f" AND owner_id = ${param_counter}"
					params.append(owner_id)
					param_counter += 1
				
				query += " ORDER BY territory_name"
				
				rows = await conn.fetch(query, *params)
				
				territories = []
				for row in rows:
					territory = self._row_to_territory(row)
					territory_data = territory.model_dump()
					
					# Add metrics if requested
					if include_metrics:
						metrics = await self._calculate_territory_metrics(territory.id, tenant_id)
						territory_data.update(metrics)
					
					territories.append(territory_data)
			
			logger.info(f"✅ Found {len(territories)} territories")
			return territories
			
		except Exception as e:
			logger.error(f"Failed to list territories: {str(e)}", exc_info=True)
			raise TerritoryError(f"List territories failed: {str(e)}")
	
	async def update_territory(
		self,
		territory_id: str,
		update_data: Dict[str, Any],
		tenant_id: str,
		updated_by: str
	) -> Territory:
		"""
		Update existing territory
		
		Args:
			territory_id: Territory identifier
			update_data: Fields to update
			tenant_id: Tenant identifier
			updated_by: User updating the territory
			
		Returns:
			Updated territory object
		"""
		try:
			# Get existing territory
			existing = await self.get_territory(territory_id, tenant_id)
			if not existing:
				raise TerritoryError(f"Territory not found: {territory_id}")
			
			# Update fields
			update_data.update({
				"updated_by": updated_by,
				"updated_at": datetime.utcnow(),
				"version": existing.version + 1
			})
			
			# Apply updates
			existing_dict = existing.model_dump()
			existing_dict.update(update_data)
			
			# Validate updated territory
			updated_territory = Territory(**existing_dict)
			
			# Save updated territory
			saved_territory = await self._save_territory(updated_territory)
			
			logger.info(f"✅ Updated territory: {territory_id}")
			return saved_territory
			
		except Exception as e:
			logger.error(f"Failed to update territory: {str(e)}", exc_info=True)
			raise TerritoryError(f"Territory update failed: {str(e)}")
	
	# ================================
	# Account Assignment
	# ================================
	
	async def assign_account_to_territory(
		self,
		account_id: str,
		territory_id: str,
		assignment_type: AssignmentType,
		tenant_id: str,
		assigned_by: str,
		assignment_reason: Optional[str] = None
	) -> AccountTerritoryAssignment:
		"""
		Assign account to territory
		
		Args:
			account_id: Account identifier
			territory_id: Territory identifier
			assignment_type: Type of assignment
			tenant_id: Tenant identifier
			assigned_by: User making the assignment
			assignment_reason: Reason for assignment
			
		Returns:
			Created assignment object
		"""
		try:
			logger.info(f"📍 Assigning account {account_id} to territory {territory_id}")
			
			# Validate account and territory exist
			account = await self.db_manager.get_account(account_id, tenant_id)
			if not account:
				raise TerritoryError(f"Account not found: {account_id}")
			
			territory = await self.get_territory(territory_id, tenant_id)
			if not territory:
				raise TerritoryError(f"Territory not found: {territory_id}")
			
			# Check for existing primary assignment if this is primary
			if assignment_type == AssignmentType.PRIMARY:
				existing_primary = await self._get_primary_assignment(account_id, tenant_id)
				if existing_primary:
					# Remove existing primary assignment
					await self._delete_assignment(existing_primary.id, tenant_id)
			
			# Create assignment
			assignment_data = {
				"tenant_id": tenant_id,
				"account_id": account_id,
				"territory_id": territory_id,
				"assignment_type": assignment_type,
				"assigned_by": assigned_by,
				"assignment_reason": assignment_reason,
				"created_by": assigned_by,
				"updated_by": assigned_by
			}
			
			assignment = AccountTerritoryAssignment(**assignment_data)
			
			# Calculate assignment score
			assignment.assignment_score = await self._calculate_assignment_score(
				account, territory
			)
			
			# Save assignment
			saved_assignment = await self._save_assignment(assignment)
			
			# Update territory metrics
			await self._update_territory_metrics(territory_id, tenant_id)
			
			logger.info(f"✅ Assigned account to territory: {saved_assignment.id}")
			return saved_assignment
			
		except Exception as e:
			logger.error(f"Failed to assign account to territory: {str(e)}", exc_info=True)
			raise TerritoryError(f"Account assignment failed: {str(e)}")
	
	async def get_account_territories(
		self,
		account_id: str,
		tenant_id: str
	) -> List[Dict[str, Any]]:
		"""
		Get all territory assignments for an account
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			
		Returns:
			List of territory assignments with details
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				rows = await conn.fetch("""
					SELECT a.*, t.territory_name, t.territory_type, t.owner_id
					FROM crm_account_territory_assignments a
					JOIN crm_territories t ON a.territory_id = t.id
					WHERE a.account_id = $1 AND a.tenant_id = $2
					ORDER BY 
						CASE a.assignment_type 
							WHEN 'primary' THEN 1 
							WHEN 'secondary' THEN 2 
							WHEN 'overlay' THEN 3 
							ELSE 4 
						END,
						a.created_at DESC
				""", account_id, tenant_id)
				
				assignments = []
				for row in rows:
					assignment = self._row_to_assignment(row)
					assignment_data = assignment.model_dump()
					assignment_data['territory_details'] = {
						"territory_name": row['territory_name'],
						"territory_type": row['territory_type'],
						"owner_id": row['owner_id']
					}
					assignments.append(assignment_data)
				
				return assignments
				
		except Exception as e:
			logger.error(f"Failed to get account territories: {str(e)}", exc_info=True)
			raise TerritoryError(f"Get account territories failed: {str(e)}")
	
	async def get_territory_accounts(
		self,
		territory_id: str,
		tenant_id: str,
		assignment_type: Optional[AssignmentType] = None,
		include_account_details: bool = True
	) -> List[Dict[str, Any]]:
		"""
		Get all accounts assigned to a territory
		
		Args:
			territory_id: Territory identifier
			tenant_id: Tenant identifier
			assignment_type: Filter by assignment type
			include_account_details: Include detailed account information
			
		Returns:
			List of account assignments with details
		"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Build query
				query = """
					SELECT a.*, acc.account_name, acc.account_type, acc.annual_revenue, acc.employee_count
					FROM crm_account_territory_assignments a
					JOIN crm_accounts acc ON a.account_id = acc.id
					WHERE a.territory_id = $1 AND a.tenant_id = $2
				"""
				params = [territory_id, tenant_id]
				
				if assignment_type:
					query += " AND a.assignment_type = $3"
					params.append(assignment_type.value)
				
				query += " ORDER BY a.assignment_type, acc.account_name"
				
				rows = await conn.fetch(query, *params)
				
				assignments = []
				for row in rows:
					assignment = self._row_to_assignment(row)
					assignment_data = assignment.model_dump()
					
					if include_account_details:
						assignment_data['account_details'] = {
							"account_name": row['account_name'],
							"account_type": row['account_type'],
							"annual_revenue": float(row['annual_revenue']) if row['annual_revenue'] else None,
							"employee_count": row['employee_count']
						}
					
					assignments.append(assignment_data)
				
				return assignments
				
		except Exception as e:
			logger.error(f"Failed to get territory accounts: {str(e)}", exc_info=True)
			raise TerritoryError(f"Get territory accounts failed: {str(e)}")
	
	# ================================
	# Territory Analytics
	# ================================
	
	async def get_territory_analytics(
		self,
		territory_id: str,
		tenant_id: str,
		start_date: Optional[datetime] = None,
		end_date: Optional[datetime] = None
	) -> TerritoryAnalytics:
		"""
		Get comprehensive territory analytics
		
		Args:
			territory_id: Territory identifier
			tenant_id: Tenant identifier
			start_date: Analysis start date
			end_date: Analysis end date
			
		Returns:
			Comprehensive territory analytics
		"""
		try:
			logger.info(f"📊 Generating territory analytics for: {territory_id}")
			
			territory = await self.get_territory(territory_id, tenant_id)
			if not territory:
				raise TerritoryError(f"Territory not found: {territory_id}")
			
			analytics = TerritoryAnalytics(
				territory_id=territory_id,
				territory_name=territory.territory_name
			)
			
			# Calculate metrics
			metrics = await self._calculate_territory_metrics(territory_id, tenant_id)
			analytics.total_accounts = metrics.get("account_count", 0)
			analytics.total_revenue = metrics.get("total_revenue", 0.0)
			analytics.quota_achievement = metrics.get("quota_achievement", 0.0)
			
			# Get account distribution
			analytics.account_distribution = await self._get_account_distribution(territory_id, tenant_id)
			
			# Get revenue distribution
			analytics.revenue_distribution = await self._get_revenue_distribution(territory_id, tenant_id)
			
			# Get performance trends
			if start_date and end_date:
				analytics.performance_trends = await self._get_performance_trends(
					territory_id, tenant_id, start_date, end_date
				)
			
			# Identify coverage gaps
			analytics.coverage_gaps = await self._identify_coverage_gaps(territory_id, tenant_id)
			
			logger.info(f"✅ Generated analytics for territory with {analytics.total_accounts} accounts")
			return analytics
			
		except Exception as e:
			logger.error(f"Territory analytics failed: {str(e)}", exc_info=True)
			raise TerritoryError(f"Territory analytics failed: {str(e)}")
	
	# ================================
	# Helper Methods
	# ================================
	
	async def _save_territory(self, territory: Territory) -> Territory:
		"""Save territory to database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_territories (
						id, tenant_id, territory_name, territory_code, territory_type, status,
						owner_id, sales_rep_ids, countries, states_provinces, cities, postal_codes,
						industries, company_size_min, company_size_max, revenue_min, revenue_max,
						product_lines, service_types, annual_quota, account_target,
						description, notes, rules, metadata, current_accounts, current_revenue,
						quota_achievement, created_at, updated_at, created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
						$17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33
					)
					ON CONFLICT (id) DO UPDATE SET
						territory_name = EXCLUDED.territory_name,
						territory_code = EXCLUDED.territory_code,
						territory_type = EXCLUDED.territory_type,
						status = EXCLUDED.status,
						owner_id = EXCLUDED.owner_id,
						sales_rep_ids = EXCLUDED.sales_rep_ids,
						countries = EXCLUDED.countries,
						states_provinces = EXCLUDED.states_provinces,
						cities = EXCLUDED.cities,
						postal_codes = EXCLUDED.postal_codes,
						industries = EXCLUDED.industries,
						company_size_min = EXCLUDED.company_size_min,
						company_size_max = EXCLUDED.company_size_max,
						revenue_min = EXCLUDED.revenue_min,
						revenue_max = EXCLUDED.revenue_max,
						product_lines = EXCLUDED.product_lines,
						service_types = EXCLUDED.service_types,
						annual_quota = EXCLUDED.annual_quota,
						account_target = EXCLUDED.account_target,
						description = EXCLUDED.description,
						notes = EXCLUDED.notes,
						rules = EXCLUDED.rules,
						metadata = EXCLUDED.metadata,
						current_accounts = EXCLUDED.current_accounts,
						current_revenue = EXCLUDED.current_revenue,
						quota_achievement = EXCLUDED.quota_achievement,
						updated_at = EXCLUDED.updated_at,
						updated_by = EXCLUDED.updated_by,
						version = EXCLUDED.version
				""",
					territory.id, territory.tenant_id, territory.territory_name,
					territory.territory_code, territory.territory_type.value,
					territory.status.value, territory.owner_id, territory.sales_rep_ids,
					territory.countries, territory.states_provinces, territory.cities,
					territory.postal_codes, territory.industries, territory.company_size_min,
					territory.company_size_max, territory.revenue_min, territory.revenue_max,
					territory.product_lines, territory.service_types, territory.annual_quota,
					territory.account_target, territory.description, territory.notes,
					territory.rules, territory.metadata, territory.current_accounts,
					territory.current_revenue, territory.quota_achievement,
					territory.created_at, territory.updated_at, territory.created_by,
					territory.updated_by, territory.version
				)
			
			return territory
			
		except Exception as e:
			logger.error(f"Failed to save territory: {str(e)}", exc_info=True)
			raise TerritoryError(f"Save territory failed: {str(e)}")
	
	async def _save_assignment(self, assignment: AccountTerritoryAssignment) -> AccountTerritoryAssignment:
		"""Save assignment to database"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_account_territory_assignments (
						id, tenant_id, account_id, territory_id, assignment_type,
						assigned_by, assignment_reason, effective_date, expiry_date,
						assignment_score, notes, metadata, created_at, updated_at,
						created_by, updated_by, version
					) VALUES (
						$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
					)
					ON CONFLICT (id) DO UPDATE SET
						assignment_type = EXCLUDED.assignment_type,
						assigned_by = EXCLUDED.assigned_by,
						assignment_reason = EXCLUDED.assignment_reason,
						effective_date = EXCLUDED.effective_date,
						expiry_date = EXCLUDED.expiry_date,
						assignment_score = EXCLUDED.assignment_score,
						notes = EXCLUDED.notes,
						metadata = EXCLUDED.metadata,
						updated_at = EXCLUDED.updated_at,
						updated_by = EXCLUDED.updated_by,
						version = EXCLUDED.version
				""",
					assignment.id, assignment.tenant_id, assignment.account_id,
					assignment.territory_id, assignment.assignment_type.value,
					assignment.assigned_by, assignment.assignment_reason,
					assignment.effective_date, assignment.expiry_date,
					assignment.assignment_score, assignment.notes, assignment.metadata,
					assignment.created_at, assignment.updated_at, assignment.created_by,
					assignment.updated_by, assignment.version
				)
			
			return assignment
			
		except Exception as e:
			logger.error(f"Failed to save assignment: {str(e)}", exc_info=True)
			raise TerritoryError(f"Save assignment failed: {str(e)}")
	
	def _row_to_territory(self, row) -> Territory:
		"""Convert database row to Territory object"""
		return Territory(
			id=row['id'],
			tenant_id=row['tenant_id'],
			territory_name=row['territory_name'],
			territory_code=row['territory_code'],
			territory_type=TerritoryType(row['territory_type']),
			status=TerritoryStatus(row['status']),
			owner_id=row['owner_id'],
			sales_rep_ids=row['sales_rep_ids'] or [],
			countries=row['countries'] or [],
			states_provinces=row['states_provinces'] or [],
			cities=row['cities'] or [],
			postal_codes=row['postal_codes'] or [],
			industries=row['industries'] or [],
			company_size_min=row['company_size_min'],
			company_size_max=row['company_size_max'],
			revenue_min=row['revenue_min'],
			revenue_max=row['revenue_max'],
			product_lines=row['product_lines'] or [],
			service_types=row['service_types'] or [],
			annual_quota=row['annual_quota'],
			account_target=row['account_target'],
			description=row['description'],
			notes=row['notes'],
			rules=row['rules'] or {},
			metadata=row['metadata'] or {},
			current_accounts=row['current_accounts'],
			current_revenue=row['current_revenue'],
			quota_achievement=row['quota_achievement'],
			created_at=row['created_at'],
			updated_at=row['updated_at'],
			created_by=row['created_by'],
			updated_by=row['updated_by'],
			version=row['version']
		)
	
	def _row_to_assignment(self, row) -> AccountTerritoryAssignment:
		"""Convert database row to AccountTerritoryAssignment object"""
		return AccountTerritoryAssignment(
			id=row['id'],
			tenant_id=row['tenant_id'],
			account_id=row['account_id'],
			territory_id=row['territory_id'],
			assignment_type=AssignmentType(row['assignment_type']),
			assigned_by=row['assigned_by'],
			assignment_reason=row['assignment_reason'],
			effective_date=row['effective_date'],
			expiry_date=row['expiry_date'],
			assignment_score=row['assignment_score'],
			notes=row['notes'],
			metadata=row['metadata'] or {},
			created_at=row['created_at'],
			updated_at=row['updated_at'],
			created_by=row['created_by'],
			updated_by=row['updated_by'],
			version=row['version']
		)
	
	async def _get_territory_by_name(self, name: str, tenant_id: str) -> Optional[Territory]:
		"""Get territory by name"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_territories 
					WHERE territory_name = $1 AND tenant_id = $2
				""", name, tenant_id)
				
				if row:
					return self._row_to_territory(row)
				return None
				
		except Exception:
			return None
	
	async def _get_primary_assignment(self, account_id: str, tenant_id: str) -> Optional[AccountTerritoryAssignment]:
		"""Get primary territory assignment for account"""
		try:
			async with self.db_manager.get_connection() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM crm_account_territory_assignments 
					WHERE account_id = $1 AND tenant_id = $2 AND assignment_type = 'primary'
				""", account_id, tenant_id)
				
				if row:
					return self._row_to_assignment(row)
				return None
				
		except Exception:
			return None
	
	async def _delete_assignment(self, assignment_id: str, tenant_id: str) -> bool:
		"""Delete assignment"""
		try:
			async with self.db_manager.get_connection() as conn:
				result = await conn.execute("""
					DELETE FROM crm_account_territory_assignments 
					WHERE id = $1 AND tenant_id = $2
				""", assignment_id, tenant_id)
				
				return result.split()[-1] == '1'
				
		except Exception:
			return False
	
	async def _calculate_assignment_score(self, account, territory: Territory) -> float:
		"""Calculate assignment quality score"""
		score = 0.0
		total_weight = 0.0
		
		# Geographic match
		if territory.countries or territory.states_provinces or territory.cities:
			# Simplified scoring - in reality would check account address
			score += 0.8 * 0.4  # 80% match * 40% weight
			total_weight += 0.4
		
		# Industry match
		if territory.industries and account.industry:
			if account.industry.lower() in [ind.lower() for ind in territory.industries]:
				score += 1.0 * 0.3  # Perfect match * 30% weight
			total_weight += 0.3
		
		# Company size match
		if territory.company_size_min or territory.company_size_max:
			if account.employee_count:
				size_match = True
				if territory.company_size_min and account.employee_count < territory.company_size_min:
					size_match = False
				if territory.company_size_max and account.employee_count > territory.company_size_max:
					size_match = False
				
				if size_match:
					score += 1.0 * 0.3  # Perfect match * 30% weight
				total_weight += 0.3
		
		return score / total_weight if total_weight > 0 else 0.5
	
	async def _calculate_territory_metrics(self, territory_id: str, tenant_id: str) -> Dict[str, Any]:
		"""Calculate territory performance metrics"""
		try:
			async with self.db_manager.get_connection() as conn:
				# Get account count and revenue
				metrics = await conn.fetchrow("""
					SELECT 
						COUNT(DISTINCT a.account_id) as account_count,
						COALESCE(SUM(acc.annual_revenue), 0) as total_revenue
					FROM crm_account_territory_assignments a
					JOIN crm_accounts acc ON a.account_id = acc.id
					WHERE a.territory_id = $1 AND a.tenant_id = $2
					AND a.assignment_type = 'primary'
				""", territory_id, tenant_id)
				
				# Get territory quota
				territory = await self.get_territory(territory_id, tenant_id)
				quota = territory.annual_quota if territory else None
				
				quota_achievement = 0.0
				if quota and quota > 0:
					quota_achievement = (float(metrics['total_revenue']) / quota) * 100
				
				return {
					"account_count": metrics['account_count'],
					"total_revenue": float(metrics['total_revenue']),
					"quota_achievement": quota_achievement
				}
				
		except Exception as e:
			logger.error(f"Failed to calculate territory metrics: {str(e)}", exc_info=True)
			return {"account_count": 0, "total_revenue": 0.0, "quota_achievement": 0.0}
	
	async def _update_territory_metrics(self, territory_id: str, tenant_id: str):
		"""Update cached territory metrics"""
		try:
			metrics = await self._calculate_territory_metrics(territory_id, tenant_id)
			
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					UPDATE crm_territories 
					SET current_accounts = $1, 
						current_revenue = $2, 
						quota_achievement = $3,
						updated_at = NOW()
					WHERE id = $4 AND tenant_id = $5
				""", 
					metrics["account_count"], 
					metrics["total_revenue"], 
					metrics["quota_achievement"],
					territory_id, 
					tenant_id
				)
				
		except Exception as e:
			logger.error(f"Failed to update territory metrics: {str(e)}", exc_info=True)
	
	async def _get_account_distribution(self, territory_id: str, tenant_id: str) -> Dict[str, int]:
		"""Get account distribution by type"""
		try:
			async with self.db_manager.get_connection() as conn:
				rows = await conn.fetch("""
					SELECT acc.account_type, COUNT(*) as count
					FROM crm_account_territory_assignments a
					JOIN crm_accounts acc ON a.account_id = acc.id
					WHERE a.territory_id = $1 AND a.tenant_id = $2
					GROUP BY acc.account_type
				""", territory_id, tenant_id)
				
				return {row['account_type']: row['count'] for row in rows}
				
		except Exception:
			return {}
	
	async def _get_revenue_distribution(self, territory_id: str, tenant_id: str) -> Dict[str, float]:
		"""Get revenue distribution by account type"""
		try:
			async with self.db_manager.get_connection() as conn:
				rows = await conn.fetch("""
					SELECT acc.account_type, COALESCE(SUM(acc.annual_revenue), 0) as revenue
					FROM crm_account_territory_assignments a
					JOIN crm_accounts acc ON a.account_id = acc.id
					WHERE a.territory_id = $1 AND a.tenant_id = $2
					GROUP BY acc.account_type
				""", territory_id, tenant_id)
				
				return {row['account_type']: float(row['revenue']) for row in rows}
				
		except Exception:
			return {}
	
	async def _get_performance_trends(
		self, 
		territory_id: str, 
		tenant_id: str, 
		start_date: datetime, 
		end_date: datetime
	) -> List[Dict[str, Any]]:
		"""Get performance trends over time"""
		# Placeholder for trend analysis
		return []
	
	async def _identify_coverage_gaps(self, territory_id: str, tenant_id: str) -> List[Dict[str, Any]]:
		"""Identify coverage gaps in territory"""
		# Placeholder for coverage gap analysis
		return []