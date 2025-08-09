"""
APG Customer Relationship Management - Account Hierarchy Management Module

Advanced account hierarchy management system for complex organizational structures,
subsidiary relationships, and territory-based account management.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from enum import Enum
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ValidationError

from .models import CRMAccount, AccountType
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class HierarchyRelationshipType(str, Enum):
	"""Types of account hierarchy relationships"""
	PARENT_CHILD = "parent_child"
	SUBSIDIARY = "subsidiary"
	DIVISION = "division"
	BRANCH = "branch"
	AFFILIATE = "affiliate"
	JOINT_VENTURE = "joint_venture"
	PARTNERSHIP = "partnership"
	ACQUISITION = "acquisition"


class AccountHierarchyNode(BaseModel):
	"""Account hierarchy node model"""
	account_id: str = Field(..., description="Account identifier")
	account_name: str = Field(..., description="Account name")
	account_type: AccountType = Field(..., description="Account type")
	parent_account_id: Optional[str] = Field(None, description="Parent account ID")
	relationship_type: Optional[HierarchyRelationshipType] = Field(None, description="Relationship type")
	level: int = Field(default=0, description="Hierarchy level (0=root)")
	path: List[str] = Field(default_factory=list, description="Path from root to this node")
	children: List['AccountHierarchyNode'] = Field(default_factory=list, description="Child nodes")
	
	# Aggregated data
	total_revenue: Optional[float] = Field(None, description="Total revenue including children")
	total_employees: Optional[int] = Field(None, description="Total employees including children")
	child_count: int = Field(default=0, description="Number of direct children")
	descendant_count: int = Field(default=0, description="Total descendants")
	
	# Metadata
	is_leaf: bool = Field(default=True, description="Whether this is a leaf node")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class HierarchyUpdateRequest(BaseModel):
	"""Request to update account hierarchy"""
	account_id: str = Field(..., description="Account to update")
	new_parent_id: Optional[str] = Field(None, description="New parent account ID")
	relationship_type: HierarchyRelationshipType = Field(default=HierarchyRelationshipType.PARENT_CHILD)
	effective_date: Optional[datetime] = Field(None, description="When change takes effect")
	notes: Optional[str] = Field(None, description="Notes about the change")


class HierarchyError(Exception):
	"""Base exception for hierarchy operations"""
	pass


class AccountHierarchyManager:
	"""
	Advanced account hierarchy management system with support for complex
	organizational structures, multiple relationship types, and territory management.
	"""
	
	def __init__(self, database_manager: DatabaseManager):
		"""
		Initialize hierarchy manager
		
		Args:
			database_manager: Database manager instance
		"""
		self.db_manager = database_manager
		self.hierarchy_cache = {}
		self.cache_ttl = 300  # 5 minutes
		self.last_cache_update = {}
	
	# ================================
	# Core Hierarchy Operations
	# ================================
	
	async def build_account_hierarchy(
		self,
		tenant_id: str,
		root_account_id: Optional[str] = None,
		max_depth: int = 10,
		include_metrics: bool = True
	) -> Dict[str, Any]:
		"""
		Build complete account hierarchy tree
		
		Args:
			tenant_id: Tenant identifier
			root_account_id: Root account to start from (None for all roots)
			max_depth: Maximum depth to traverse
			include_metrics: Whether to include aggregated metrics
			
		Returns:
			Hierarchy tree with all nodes and relationships
		"""
		try:
			logger.info(f"🌳 Building account hierarchy for tenant: {tenant_id}")
			
			# Get all accounts for this tenant
			accounts_result = await self.db_manager.list_accounts(tenant_id, limit=5000)
			accounts = accounts_result.get('items', [])
			
			if not accounts:
				return {
					"hierarchy": [],
					"total_accounts": 0,
					"max_depth": 0,
					"root_nodes": 0
				}
			
			# Build account lookup map
			account_map = {acc.id: acc for acc in accounts}
			
			# Build hierarchy structure
			hierarchy_nodes = {}
			root_nodes = []
			
			# First pass: create all nodes
			for account in accounts:
				node = AccountHierarchyNode(
					account_id=account.id,
					account_name=account.account_name,
					account_type=account.account_type,
					parent_account_id=account.parent_account_id,
					relationship_type=HierarchyRelationshipType.PARENT_CHILD
				)
				hierarchy_nodes[account.id] = node
			
			# Second pass: build parent-child relationships
			for account in accounts:
				node = hierarchy_nodes[account.id]
				
				if account.parent_account_id and account.parent_account_id in hierarchy_nodes:
					# Add to parent's children
					parent_node = hierarchy_nodes[account.parent_account_id]
					parent_node.children.append(node)
					parent_node.is_leaf = False
					parent_node.child_count += 1
				else:
					# This is a root node
					root_nodes.append(node)
			
			# Third pass: calculate levels, paths, and descendant counts
			await self._calculate_hierarchy_metrics(root_nodes, hierarchy_nodes, max_depth)
			
			# Include aggregated metrics if requested
			if include_metrics:
				await self._calculate_aggregated_metrics(hierarchy_nodes, accounts)
			
			# Filter by root account if specified
			if root_account_id:
				if root_account_id in hierarchy_nodes:
					root_nodes = [hierarchy_nodes[root_account_id]]
				else:
					raise HierarchyError(f"Root account not found: {root_account_id}")
			
			# Calculate hierarchy statistics
			total_depth = max((node.level for node in hierarchy_nodes.values()), default=0)
			
			return {
				"hierarchy": [node.model_dump() for node in root_nodes],
				"total_accounts": len(accounts),
				"max_depth": total_depth,
				"root_nodes": len(root_nodes),
				"hierarchy_map": {node_id: node.model_dump() for node_id, node in hierarchy_nodes.items()},
				"generated_at": datetime.utcnow().isoformat()
			}
			
		except Exception as e:
			logger.error(f"Failed to build account hierarchy: {str(e)}", exc_info=True)
			raise HierarchyError(f"Hierarchy build failed: {str(e)}")
	
	async def get_account_ancestors(
		self,
		account_id: str,
		tenant_id: str,
		include_self: bool = False
	) -> List[Dict[str, Any]]:
		"""
		Get all ancestor accounts up the hierarchy
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			include_self: Whether to include the account itself
			
		Returns:
			List of ancestor accounts from immediate parent to root
		"""
		try:
			ancestors = []
			current_id = account_id
			visited = set()
			
			# Get account to start traversal
			if include_self:
				current_account = await self.db_manager.get_account(account_id, tenant_id)
				if current_account:
					ancestors.append({
						"account_id": current_account.id,
						"account_name": current_account.account_name,
						"account_type": current_account.account_type,
						"level": 0
					})
					current_id = current_account.parent_account_id
				else:
					raise HierarchyError(f"Account not found: {account_id}")
			else:
				account = await self.db_manager.get_account(account_id, tenant_id)
				if account:
					current_id = account.parent_account_id
				else:
					raise HierarchyError(f"Account not found: {account_id}")
			
			level = 1 if include_self else 0
			
			# Traverse up the hierarchy
			while current_id and current_id not in visited:
				visited.add(current_id)
				
				parent_account = await self.db_manager.get_account(current_id, tenant_id)
				if parent_account:
					ancestors.append({
						"account_id": parent_account.id,
						"account_name": parent_account.account_name,
						"account_type": parent_account.account_type,
						"level": level,
						"relationship": "parent"
					})
					current_id = parent_account.parent_account_id
					level += 1
				else:
					break
				
				# Prevent infinite loops
				if level > 20:
					logger.warning(f"Hierarchy depth exceeded for account {account_id}")
					break
			
			return ancestors
			
		except Exception as e:
			logger.error(f"Failed to get account ancestors: {str(e)}", exc_info=True)
			raise HierarchyError(f"Get ancestors failed: {str(e)}")
	
	async def get_account_descendants(
		self,
		account_id: str,
		tenant_id: str,
		max_depth: int = 5,
		include_self: bool = False
	) -> List[Dict[str, Any]]:
		"""
		Get all descendant accounts down the hierarchy
		
		Args:
			account_id: Account identifier
			tenant_id: Tenant identifier
			max_depth: Maximum depth to traverse
			include_self: Whether to include the account itself
			
		Returns:
			List of descendant accounts with hierarchy information
		"""
		try:
			descendants = []
			
			# Include self if requested
			if include_self:
				root_account = await self.db_manager.get_account(account_id, tenant_id)
				if root_account:
					descendants.append({
						"account_id": root_account.id,
						"account_name": root_account.account_name,
						"account_type": root_account.account_type,
						"level": 0,
						"relationship": "self"
					})
				else:
					raise HierarchyError(f"Account not found: {account_id}")
			
			# Get descendants using BFS
			queue = [(account_id, 0)]
			visited = set([account_id])
			
			while queue:
				current_id, current_level = queue.pop(0)
				
				if current_level >= max_depth:
					continue
				
				# Get direct children
				children = await self._get_direct_children(current_id, tenant_id)
				
				for child in children:
					if child.id not in visited:
						visited.add(child.id)
						
						descendants.append({
							"account_id": child.id,
							"account_name": child.account_name,
							"account_type": child.account_type,
							"level": current_level + 1,
							"parent_account_id": current_id,
							"relationship": "descendant"
						})
						
						# Add to queue for further exploration
						queue.append((child.id, current_level + 1))
			
			return descendants
			
		except Exception as e:
			logger.error(f"Failed to get account descendants: {str(e)}", exc_info=True)
			raise HierarchyError(f"Get descendants failed: {str(e)}")
	
	async def update_account_hierarchy(
		self,
		update_request: HierarchyUpdateRequest,
		tenant_id: str,
		updated_by: str
	) -> Dict[str, Any]:
		"""
		Update account hierarchy relationships
		
		Args:
			update_request: Hierarchy update request
			tenant_id: Tenant identifier
			updated_by: User making the change
			
		Returns:
			Update result with validation information
		"""
		try:
			logger.info(f"🔄 Updating hierarchy for account: {update_request.account_id}")
			
			# Validate the update request
			validation_result = await self._validate_hierarchy_update(
				update_request, tenant_id
			)
			
			if not validation_result["valid"]:
				raise HierarchyError(f"Invalid hierarchy update: {validation_result['reason']}")
			
			# Get the account to update
			account = await self.db_manager.get_account(update_request.account_id, tenant_id)
			if not account:
				raise HierarchyError(f"Account not found: {update_request.account_id}")
			
			# Store old parent for audit trail
			old_parent_id = account.parent_account_id
			
			# Update the account's parent
			update_data = {
				"parent_account_id": update_request.new_parent_id,
				"updated_by": updated_by,
				"updated_at": datetime.utcnow()
			}
			
			updated_account = await self.db_manager.update_account(
				update_request.account_id, update_data, tenant_id, updated_by
			)
			
			# Record hierarchy change in audit log
			await self._record_hierarchy_change(
				account_id=update_request.account_id,
				old_parent_id=old_parent_id,
				new_parent_id=update_request.new_parent_id,
				relationship_type=update_request.relationship_type,
				tenant_id=tenant_id,
				updated_by=updated_by,
				notes=update_request.notes
			)
			
			# Clear hierarchy cache
			self._clear_hierarchy_cache(tenant_id)
			
			return {
				"success": True,
				"account_id": update_request.account_id,
				"old_parent_id": old_parent_id,
				"new_parent_id": update_request.new_parent_id,
				"relationship_type": update_request.relationship_type,
				"updated_at": datetime.utcnow().isoformat(),
				"validation": validation_result
			}
			
		except Exception as e:
			logger.error(f"Failed to update account hierarchy: {str(e)}", exc_info=True)
			raise HierarchyError(f"Hierarchy update failed: {str(e)}")
	
	# ================================
	# Hierarchy Analytics
	# ================================
	
	async def get_hierarchy_analytics(
		self,
		tenant_id: str,
		account_id: Optional[str] = None
	) -> Dict[str, Any]:
		"""
		Get hierarchy analytics and metrics
		
		Args:
			tenant_id: Tenant identifier
			account_id: Specific account to analyze (None for all)
			
		Returns:
			Comprehensive hierarchy analytics
		"""
		try:
			analytics = {
				"tenant_id": tenant_id,
				"analyzed_at": datetime.utcnow().isoformat(),
				"hierarchy_metrics": {},
				"account_distribution": {},
				"depth_analysis": {},
				"relationship_analysis": {}
			}
			
			# Build hierarchy to get metrics
			hierarchy_data = await self.build_account_hierarchy(
				tenant_id, account_id, include_metrics=True
			)
			
			analytics["hierarchy_metrics"] = {
				"total_accounts": hierarchy_data["total_accounts"],
				"max_depth": hierarchy_data["max_depth"],
				"root_nodes": hierarchy_data["root_nodes"],
				"average_children_per_node": 0,
				"leaf_nodes": 0
			}
			
			# Analyze hierarchy structure
			if hierarchy_data["hierarchy_map"]:
				nodes = hierarchy_data["hierarchy_map"].values()
				
				# Calculate averages
				total_children = sum(node.get("child_count", 0) for node in nodes)
				non_leaf_nodes = sum(1 for node in nodes if not node.get("is_leaf", True))
				
				if non_leaf_nodes > 0:
					analytics["hierarchy_metrics"]["average_children_per_node"] = total_children / non_leaf_nodes
				
				analytics["hierarchy_metrics"]["leaf_nodes"] = sum(
					1 for node in nodes if node.get("is_leaf", True)
				)
				
				# Account type distribution
				type_distribution = {}
				for node in nodes:
					account_type = node.get("account_type", "unknown")
					type_distribution[account_type] = type_distribution.get(account_type, 0) + 1
				
				analytics["account_distribution"] = type_distribution
				
				# Depth analysis
				depth_distribution = {}
				for node in nodes:
					level = node.get("level", 0)
					depth_distribution[f"level_{level}"] = depth_distribution.get(f"level_{level}", 0) + 1
				
				analytics["depth_analysis"] = depth_distribution
			
			return analytics
			
		except Exception as e:
			logger.error(f"Hierarchy analytics failed: {str(e)}", exc_info=True)
			raise HierarchyError(f"Hierarchy analytics failed: {str(e)}")
	
	async def find_hierarchy_path(
		self,
		from_account_id: str,
		to_account_id: str,
		tenant_id: str
	) -> Optional[List[Dict[str, Any]]]:
		"""
		Find path between two accounts in hierarchy
		
		Args:
			from_account_id: Starting account
			to_account_id: Target account
			tenant_id: Tenant identifier
			
		Returns:
			Path between accounts or None if no path exists
		"""
		try:
			# Get ancestors of both accounts
			from_ancestors = await self.get_account_ancestors(
				from_account_id, tenant_id, include_self=True
			)
			to_ancestors = await self.get_account_ancestors(
				to_account_id, tenant_id, include_self=True
			)
			
			# Create ancestor sets for quick lookup
			from_ancestor_ids = {acc["account_id"] for acc in from_ancestors}
			to_ancestor_ids = {acc["account_id"] for acc in to_ancestors}
			
			# Find common ancestors
			common_ancestors = from_ancestor_ids & to_ancestor_ids
			
			if not common_ancestors:
				return None  # No path exists
			
			# Find the lowest common ancestor (LCA)
			lca_id = None
			min_level = float('inf')
			
			for ancestor_id in common_ancestors:
				from_level = next(acc["level"] for acc in from_ancestors if acc["account_id"] == ancestor_id)
				to_level = next(acc["level"] for acc in to_ancestors if acc["account_id"] == ancestor_id)
				max_level = max(from_level, to_level)
				
				if max_level < min_level:
					min_level = max_level
					lca_id = ancestor_id
			
			if not lca_id:
				return None
			
			# Build path: from_account -> LCA -> to_account
			path = []
			
			# Path from from_account to LCA
			for ancestor in from_ancestors:
				path.append({
					"account_id": ancestor["account_id"],
					"account_name": ancestor["account_name"],
					"direction": "up"
				})
				if ancestor["account_id"] == lca_id:
					break
			
			# Path from LCA to to_account (excluding LCA itself)
			lca_to_target = []
			for ancestor in to_ancestors:
				if ancestor["account_id"] == lca_id:
					break
				lca_to_target.append({
					"account_id": ancestor["account_id"],
					"account_name": ancestor["account_name"],
					"direction": "down"
				})
			
			# Reverse and add to path
			lca_to_target.reverse()
			path.extend(lca_to_target)
			
			return path
			
		except Exception as e:
			logger.error(f"Find hierarchy path failed: {str(e)}", exc_info=True)
			raise HierarchyError(f"Find hierarchy path failed: {str(e)}")
	
	# ================================
	# Helper Methods
	# ================================
	
	async def _calculate_hierarchy_metrics(
		self,
		root_nodes: List[AccountHierarchyNode],
		hierarchy_nodes: Dict[str, AccountHierarchyNode],
		max_depth: int
	):
		"""Calculate hierarchy levels, paths, and descendant counts"""
		def calculate_node_metrics(node: AccountHierarchyNode, level: int, path: List[str]):
			node.level = level
			node.path = path + [node.account_id]
			
			descendant_count = 0
			for child in node.children:
				if level < max_depth:
					calculate_node_metrics(child, level + 1, node.path)
					descendant_count += 1 + child.descendant_count
			
			node.descendant_count = descendant_count
		
		# Calculate metrics for each root node
		for root_node in root_nodes:
			calculate_node_metrics(root_node, 0, [])
	
	async def _calculate_aggregated_metrics(
		self,
		hierarchy_nodes: Dict[str, AccountHierarchyNode],
		accounts: List[CRMAccount]
	):
		"""Calculate aggregated financial and employee metrics"""
		account_data = {acc.id: acc for acc in accounts}
		
		def aggregate_metrics(node: AccountHierarchyNode):
			account = account_data.get(node.account_id)
			
			# Start with own metrics
			total_revenue = float(account.annual_revenue) if account and account.annual_revenue else 0
			total_employees = account.employee_count if account and account.employee_count else 0
			
			# Add children metrics
			for child in node.children:
				aggregate_metrics(child)
				total_revenue += child.total_revenue or 0
				total_employees += child.total_employees or 0
			
			node.total_revenue = total_revenue
			node.total_employees = total_employees
		
		# Calculate for all root nodes
		for node in hierarchy_nodes.values():
			if node.level == 0:  # Root node
				aggregate_metrics(node)
	
	async def _get_direct_children(
		self,
		parent_account_id: str,
		tenant_id: str
	) -> List[CRMAccount]:
		"""Get direct child accounts"""
		try:
			async with self.db_manager.get_connection() as conn:
				rows = await conn.fetch("""
					SELECT * FROM crm_accounts 
					WHERE parent_account_id = $1 AND tenant_id = $2
					AND status = 'active'
					ORDER BY account_name
				""", parent_account_id, tenant_id)
				
				return [self.db_manager._row_to_account(row) for row in rows]
				
		except Exception as e:
			logger.error(f"Failed to get direct children: {str(e)}", exc_info=True)
			return []
	
	async def _validate_hierarchy_update(
		self,
		update_request: HierarchyUpdateRequest,
		tenant_id: str
	) -> Dict[str, Any]:
		"""Validate hierarchy update to prevent cycles and invalid relationships"""
		try:
			# Check if account exists
			account = await self.db_manager.get_account(update_request.account_id, tenant_id)
			if not account:
				return {"valid": False, "reason": "Account not found"}
			
			# Check if new parent exists (if specified)
			if update_request.new_parent_id:
				parent_account = await self.db_manager.get_account(update_request.new_parent_id, tenant_id)
				if not parent_account:
					return {"valid": False, "reason": "Parent account not found"}
				
				# Check for self-assignment
				if update_request.account_id == update_request.new_parent_id:
					return {"valid": False, "reason": "Account cannot be its own parent"}
				
				# Check for circular reference
				ancestors = await self.get_account_ancestors(
					update_request.new_parent_id, tenant_id, include_self=True
				)
				
				ancestor_ids = {acc["account_id"] for acc in ancestors}
				if update_request.account_id in ancestor_ids:
					return {"valid": False, "reason": "Circular reference detected"}
			
			return {"valid": True, "reason": "Validation passed"}
			
		except Exception as e:
			logger.error(f"Hierarchy validation failed: {str(e)}", exc_info=True)
			return {"valid": False, "reason": f"Validation error: {str(e)}"}
	
	async def _record_hierarchy_change(
		self,
		account_id: str,
		old_parent_id: Optional[str],
		new_parent_id: Optional[str],
		relationship_type: HierarchyRelationshipType,
		tenant_id: str,
		updated_by: str,
		notes: Optional[str] = None
	):
		"""Record hierarchy change in audit log"""
		try:
			async with self.db_manager.get_connection() as conn:
				await conn.execute("""
					INSERT INTO crm_hierarchy_audit_log (
						id, tenant_id, account_id, old_parent_id, new_parent_id,
						relationship_type, change_notes, created_by, created_at
					) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
				""", 
					uuid7str(), tenant_id, account_id, old_parent_id, new_parent_id,
					relationship_type.value, notes, updated_by, datetime.utcnow()
				)
				
		except Exception as e:
			logger.error(f"Failed to record hierarchy change: {str(e)}", exc_info=True)
			# Don't raise error - audit logging shouldn't block operation
	
	def _clear_hierarchy_cache(self, tenant_id: str):
		"""Clear hierarchy cache for tenant"""
		cache_key = f"hierarchy_{tenant_id}"
		if cache_key in self.hierarchy_cache:
			del self.hierarchy_cache[cache_key]
		if cache_key in self.last_cache_update:
			del self.last_cache_update[cache_key]