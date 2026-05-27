"""
APG Capability Registry - Mobile and Offline Service

Mobile-optimized service layer with offline capabilities, progressive web app support,
and mobile-specific optimizations for the capability registry.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from .models import CRCapability, CRComposition, CRRegistry
from .service import CRService

# =============================================================================
# Mobile-Optimized Data Models
# =============================================================================

class MobileCapabilityView(BaseModel):
	"""Lightweight capability view for mobile devices."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	capability_id: str = Field(..., description="Capability ID")
	code: str = Field(..., description="Capability code")
	name: str = Field(..., description="Display name")
	description: str = Field(..., description="Short description")
	category: str = Field(..., description="Category")
	version: str = Field(..., description="Version")
	quality_score: float = Field(0.0, ge=0.0, le=1.0)
	status: str = Field(..., description="Status")
	last_sync: Optional[datetime] = Field(None, description="Last sync time")

class MobileCompositionView(BaseModel):
	"""Lightweight composition view for mobile devices."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	composition_id: str = Field(..., description="Composition ID")
	name: str = Field(..., description="Name")
	description: str = Field(..., description="Description")
	type: str = Field(..., description="Type")
	capability_count: int = Field(0, ge=0)
	validation_score: float = Field(0.0, ge=0.0, le=1.0)
	is_offline_ready: bool = Field(False, description="Offline ready")
	last_sync: Optional[datetime] = Field(None, description="Last sync time")

class OfflineAction(BaseModel):
	"""Model for offline actions queue."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	action_id: str = Field(default_factory=uuid7str)
	action_type: str = Field(..., description="Action type")
	resource_type: str = Field(..., description="Resource type")
	resource_id: str = Field(..., description="Resource ID")
	payload: Dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	retry_count: int = Field(0, ge=0)
	max_retries: int = Field(3, ge=0)
	status: str = Field("pending", description="Action status")

class SyncManifest(BaseModel):
	"""Sync manifest for offline data management."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)

	manifest_id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Tenant ID")
	last_sync: datetime = Field(default_factory=datetime.utcnow)
	capabilities_count: int = Field(0, ge=0)
	compositions_count: int = Field(0, ge=0)
	offline_actions_count: int = Field(0, ge=0)
	sync_version: str = Field("1.0", description="Sync version")
	checksum: str = Field(..., description="Data checksum")

# =============================================================================
# Mobile and Offline Service
# =============================================================================

class MobileOfflineService:
	"""Service for mobile and offline capabilities."""

	def __init__(
		self,
		tenant_id: str = "default",
		offline_db_path: Optional[str] = None
	):
		self.tenant_id = tenant_id
		self.offline_db_path = offline_db_path or f"/tmp/apg_cr_offline_{tenant_id}.db"
		self.cr_service: Optional[CRService] = None
		self.offline_actions: List[OfflineAction] = []
		self.is_online = True
		self.last_sync: Optional[datetime] = None

		self._init_offline_db()

	def _init_offline_db(self):
		"""Initialize SQLite offline database."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		# Capabilities table
		cursor.execute("""
			CREATE TABLE IF NOT EXISTS offline_capabilities (
				capability_id TEXT PRIMARY KEY,
				code TEXT NOT NULL,
				name TEXT NOT NULL,
				description TEXT,
				category TEXT,
				version TEXT,
				quality_score REAL DEFAULT 0.0,
				status TEXT,
				data_json TEXT,
				last_sync TIMESTAMP,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
		""")

		# Compositions table
		cursor.execute("""
			CREATE TABLE IF NOT EXISTS offline_compositions (
				composition_id TEXT PRIMARY KEY,
				name TEXT NOT NULL,
				description TEXT,
				type TEXT,
				capability_count INTEGER DEFAULT 0,
				validation_score REAL DEFAULT 0.0,
				is_offline_ready BOOLEAN DEFAULT FALSE,
				data_json TEXT,
				last_sync TIMESTAMP,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)
		""")

		# Offline actions queue
		cursor.execute("""
			CREATE TABLE IF NOT EXISTS offline_actions (
				action_id TEXT PRIMARY KEY,
				action_type TEXT NOT NULL,
				resource_type TEXT NOT NULL,
				resource_id TEXT NOT NULL,
				payload_json TEXT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
				retry_count INTEGER DEFAULT 0,
				max_retries INTEGER DEFAULT 3,
				status TEXT DEFAULT 'pending'
			)
		""")

		# Sync manifest
		cursor.execute("""
			CREATE TABLE IF NOT EXISTS sync_manifest (
				tenant_id TEXT PRIMARY KEY,
				last_sync TIMESTAMP,
				capabilities_count INTEGER DEFAULT 0,
				compositions_count INTEGER DEFAULT 0,
				offline_actions_count INTEGER DEFAULT 0,
				sync_version TEXT DEFAULT '1.0',
				checksum TEXT
			)
		""")

		# Create indexes
		cursor.execute("CREATE INDEX IF NOT EXISTS idx_capabilities_category ON offline_capabilities(category)")
		cursor.execute("CREATE INDEX IF NOT EXISTS idx_capabilities_status ON offline_capabilities(status)")
		cursor.execute("CREATE INDEX IF NOT EXISTS idx_compositions_type ON offline_compositions(type)")
		cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_status ON offline_actions(status)")

		conn.commit()
		conn.close()

	async def set_online_service(self, cr_service: CRService):
		"""Set the online service instance."""
		self.cr_service = cr_service

	async def set_connection_status(self, is_online: bool):
		"""Set connection status."""
		self.is_online = is_online
		if is_online and self.cr_service:
			await self.sync_offline_actions()

	# =========================================================================
	# Mobile-Optimized Capability Operations
	# =========================================================================

	async def get_mobile_capabilities(
		self,
		category: Optional[str] = None,
		limit: int = 50,
		offset: int = 0
	) -> List[MobileCapabilityView]:
		"""Get mobile-optimized capability list."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		query = """
			SELECT capability_id, code, name, description, category,
				   version, quality_score, status, last_sync
			FROM offline_capabilities
		"""
		params = []

		if category:
			query += " WHERE category = ?"
			params.append(category)

		query += " ORDER BY name LIMIT ? OFFSET ?"
		params.extend([limit, offset])

		cursor.execute(query, params)
		rows = cursor.fetchall()
		conn.close()

		capabilities = []
		for row in rows:
			capabilities.append(MobileCapabilityView(
				capability_id=row[0],
				code=row[1],
				name=row[2],
				description=row[3] or "",
				category=row[4],
				version=row[5],
				quality_score=row[6],
				status=row[7],
				last_sync=datetime.fromisoformat(row[8]) if row[8] else None
			))

		return capabilities

	async def get_mobile_capability_detail(
		self,
		capability_id: str
	) -> Optional[Dict[str, Any]]:
		"""Get detailed capability info for mobile."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		cursor.execute(
			"SELECT data_json FROM offline_capabilities WHERE capability_id = ?",
			(capability_id,)
		)
		row = cursor.fetchone()
		conn.close()

		if row and row[0]:
			return json.loads(row[0])
		return None

	async def search_mobile_capabilities(
		self,
		query: str,
		category: Optional[str] = None,
		limit: int = 20
	) -> List[MobileCapabilityView]:
		"""Search capabilities with mobile optimization."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		sql_query = """
			SELECT capability_id, code, name, description, category,
				   version, quality_score, status, last_sync
			FROM offline_capabilities
			WHERE (name LIKE ? OR code LIKE ? OR description LIKE ?)
		"""
		params = [f"%{query}%", f"%{query}%", f"%{query}%"]

		if category:
			sql_query += " AND category = ?"
			params.append(category)

		sql_query += " ORDER BY quality_score DESC LIMIT ?"
		params.append(limit)

		cursor.execute(sql_query, params)
		rows = cursor.fetchall()
		conn.close()

		capabilities = []
		for row in rows:
			capabilities.append(MobileCapabilityView(
				capability_id=row[0],
				code=row[1],
				name=row[2],
				description=row[3] or "",
				category=row[4],
				version=row[5],
				quality_score=row[6],
				status=row[7],
				last_sync=datetime.fromisoformat(row[8]) if row[8] else None
			))

		return capabilities

	# =========================================================================
	# Mobile-Optimized Composition Operations
	# =========================================================================

	async def get_mobile_compositions(
		self,
		composition_type: Optional[str] = None,
		limit: int = 50,
		offset: int = 0
	) -> List[MobileCompositionView]:
		"""Get mobile-optimized composition list."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		query = """
			SELECT composition_id, name, description, type, capability_count,
				   validation_score, is_offline_ready, last_sync
			FROM offline_compositions
		"""
		params = []

		if composition_type:
			query += " WHERE type = ?"
			params.append(composition_type)

		query += " ORDER BY name LIMIT ? OFFSET ?"
		params.extend([limit, offset])

		cursor.execute(query, params)
		rows = cursor.fetchall()
		conn.close()

		compositions = []
		for row in rows:
			compositions.append(MobileCompositionView(
				composition_id=row[0],
				name=row[1],
				description=row[2] or "",
				type=row[3],
				capability_count=row[4],
				validation_score=row[5],
				is_offline_ready=bool(row[6]),
				last_sync=datetime.fromisoformat(row[7]) if row[7] else None
			))

		return compositions

	async def create_mobile_composition(
		self,
		name: str,
		description: str,
		capability_ids: List[str],
		composition_type: str = "mobile"
	) -> str:
		"""Create composition optimized for mobile use."""
		composition_id = uuid7str()

		# Store offline action if not online
		if not self.is_online:
			action = OfflineAction(
				action_type="create_composition",
				resource_type="composition",
				resource_id=composition_id,
				payload={
					"name": name,
					"description": description,
					"capability_ids": capability_ids,
					"composition_type": composition_type
				}
			)
			await self._store_offline_action(action)

		# Store in offline database
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		composition_data = {
			"name": name,
			"description": description,
			"capability_ids": capability_ids,
			"composition_type": composition_type,
			"created_offline": not self.is_online
		}

		cursor.execute("""
			INSERT INTO offline_compositions
			(composition_id, name, description, type, capability_count,
			 validation_score, is_offline_ready, data_json, last_sync)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		""", (
			composition_id, name, description, composition_type,
			len(capability_ids), 0.8, True, json.dumps(composition_data),
			datetime.utcnow().isoformat()
		))

		conn.commit()
		conn.close()

		return composition_id

	# =========================================================================
	# Offline Actions Management
	# =========================================================================

	async def _store_offline_action(self, action: OfflineAction):
		"""Store offline action in queue."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		cursor.execute("""
			INSERT INTO offline_actions
			(action_id, action_type, resource_type, resource_id,
			 payload_json, created_at, retry_count, max_retries, status)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		""", (
			action.action_id, action.action_type, action.resource_type,
			action.resource_id, json.dumps(action.payload),
			action.created_at.isoformat(), action.retry_count,
			action.max_retries, action.status
		))

		conn.commit()
		conn.close()

	async def get_pending_offline_actions(self) -> List[OfflineAction]:
		"""Get pending offline actions."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		cursor.execute("""
			SELECT action_id, action_type, resource_type, resource_id,
				   payload_json, created_at, retry_count, max_retries, status
			FROM offline_actions
			WHERE status = 'pending'
			ORDER BY created_at
		""")

		rows = cursor.fetchall()
		conn.close()

		actions = []
		for row in rows:
			actions.append(OfflineAction(
				action_id=row[0],
				action_type=row[1],
				resource_type=row[2],
				resource_id=row[3],
				payload=json.loads(row[4]) if row[4] else {},
				created_at=datetime.fromisoformat(row[5]),
				retry_count=row[6],
				max_retries=row[7],
				status=row[8]
			))

		return actions

	async def sync_offline_actions(self) -> Dict[str, Any]:
		"""Sync pending offline actions with online service."""
		if not self.is_online or not self.cr_service:
			return {"synced": 0, "failed": 0, "message": "Offline mode"}

		actions = await self.get_pending_offline_actions()
		synced_count = 0
		failed_count = 0

		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		for action in actions:
			try:
				# Process different action types
				if action.action_type == "create_composition":
					await self._sync_create_composition(action)
				elif action.action_type == "update_capability":
					await self._sync_update_capability(action)
				# Add more action types as needed

				# Mark as completed
				cursor.execute(
					"UPDATE offline_actions SET status = 'completed' WHERE action_id = ?",
					(action.action_id,)
				)
				synced_count += 1

			except Exception as e:
				# Mark as failed or retry
				new_retry_count = action.retry_count + 1
				if new_retry_count >= action.max_retries:
					cursor.execute(
						"UPDATE offline_actions SET status = 'failed', retry_count = ? WHERE action_id = ?",
						(new_retry_count, action.action_id)
					)
				else:
					cursor.execute(
						"UPDATE offline_actions SET retry_count = ? WHERE action_id = ?",
						(new_retry_count, action.action_id)
					)
				failed_count += 1

		conn.commit()
		conn.close()

		return {
			"synced": synced_count,
			"failed": failed_count,
			"message": f"Synced {synced_count} actions, {failed_count} failed"
		}

	async def _sync_create_composition(self, action: OfflineAction):
		"""Sync create composition action."""
		payload = action.payload

		# Create composition through online service
		composition_data = {
			"name": payload["name"],
			"description": payload["description"],
			"composition_type": payload["composition_type"],
			"capability_ids": payload["capability_ids"]
		}

		# This would call the actual online service
		# result = await self.cr_service.create_composition(composition_data)

		# For now, just mark as synced
		pass

	async def _sync_update_capability(self, action: OfflineAction):
		"""Sync update capability action."""
		# Implementation for syncing capability updates
		pass

	# =========================================================================
	# Data Synchronization
	# =========================================================================

	async def sync_from_online(
		self,
		force_full_sync: bool = False
	) -> Dict[str, Any]:
		"""Sync data from online service to offline storage."""
		if not self.is_online or not self.cr_service:
			return {"success": False, "message": "Offline mode"}

		try:
			sync_start = datetime.utcnow()

			# Get sync manifest
			manifest = await self._get_sync_manifest()

			# Determine sync strategy
			if force_full_sync or not manifest or not self.last_sync:
				# Full sync
				capabilities_synced = await self._full_sync_capabilities()
				compositions_synced = await self._full_sync_compositions()
			else:
				# Incremental sync
				capabilities_synced = await self._incremental_sync_capabilities(self.last_sync)
				compositions_synced = await self._incremental_sync_compositions(self.last_sync)

			# Update sync manifest
			await self._update_sync_manifest(sync_start, capabilities_synced, compositions_synced)

			self.last_sync = sync_start

			return {
				"success": True,
				"capabilities_synced": capabilities_synced,
				"compositions_synced": compositions_synced,
				"sync_time": (datetime.utcnow() - sync_start).total_seconds()
			}

		except Exception as e:
			return {
				"success": False,
				"message": f"Sync failed: {str(e)}"
			}

	async def _full_sync_capabilities(self) -> int:
		"""Perform full sync of capabilities."""
		capabilities = await self._fetch_online_capabilities()
		return await self._store_synced_capabilities(capabilities, full_sync=True)

	async def _full_sync_compositions(self) -> int:
		"""Perform full sync of compositions."""
		compositions = await self._fetch_online_compositions()
		return await self._store_synced_compositions(compositions, full_sync=True)

	async def _incremental_sync_capabilities(self, since: datetime) -> int:
		"""Perform incremental sync of capabilities."""
		capabilities = await self._fetch_online_capabilities(since)
		return await self._store_synced_capabilities(capabilities, full_sync=False)

	async def _incremental_sync_compositions(self, since: datetime) -> int:
		"""Perform incremental sync of compositions."""
		compositions = await self._fetch_online_compositions(since)
		return await self._store_synced_compositions(compositions, full_sync=False)

	async def _fetch_online_capabilities(self, since: Optional[datetime] = None) -> List[Any]:
		"""Fetch capabilities from the configured online registry service."""
		if not self.cr_service:
			return []

		if hasattr(self.cr_service, "search_capabilities"):
			result = await self.cr_service.search_capabilities(limit=1000, offset=0)
			capabilities = result.get("capabilities", []) if isinstance(result, dict) else []
		elif hasattr(self.cr_service, "list_capabilities"):
			result = await self.cr_service.list_capabilities(limit=1000, offset=0)
			capabilities = result.get("capabilities", result) if isinstance(result, dict) else result
		else:
			capabilities = getattr(self.cr_service, "capabilities", [])

		return [
			capability for capability in list(capabilities or [])
			if self._record_updated_since(capability, since)
		]

	async def _fetch_online_compositions(self, since: Optional[datetime] = None) -> List[Any]:
		"""Fetch compositions from the configured online registry service."""
		if not self.cr_service:
			return []

		if hasattr(self.cr_service, "search_compositions"):
			result = await self.cr_service.search_compositions({"limit": 1000, "offset": 0})
			compositions = result.get("compositions", result) if isinstance(result, dict) else result
		elif hasattr(self.cr_service, "list_compositions"):
			result = await self.cr_service.list_compositions(limit=1000, offset=0)
			compositions = result.get("compositions", result) if isinstance(result, dict) else result
		elif hasattr(self.cr_service, "compositions"):
			compositions = getattr(self.cr_service, "compositions", [])
		elif hasattr(self.cr_service, "db_session"):
			result = await self.cr_service.db_session.execute(
				select(CRComposition).where(CRComposition.tenant_id == self.tenant_id)
			)
			compositions = result.scalars().all()
		else:
			compositions = []

		return [
			composition for composition in list(compositions or [])
			if self._record_updated_since(composition, since)
		]

	async def _store_synced_capabilities(self, capabilities: List[Any], full_sync: bool) -> int:
		"""Persist synced capabilities to offline storage."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		if full_sync:
			cursor.execute("DELETE FROM offline_capabilities")

		synced_at = datetime.utcnow().isoformat()
		for capability in capabilities:
			record = self._normalize_capability(capability)
			cursor.execute("""
				INSERT OR REPLACE INTO offline_capabilities
				(capability_id, code, name, description, category, version,
				 quality_score, status, data_json, last_sync)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			""", (
				record["capability_id"], record["code"], record["name"],
				record["description"], record["category"], record["version"],
				record["quality_score"], record["status"],
				json.dumps(record["data"], default=str), synced_at
			))

		conn.commit()
		conn.close()

		return len(capabilities)

	async def _store_synced_compositions(self, compositions: List[Any], full_sync: bool) -> int:
		"""Persist synced compositions to offline storage."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		if full_sync:
			cursor.execute("DELETE FROM offline_compositions")

		synced_at = datetime.utcnow().isoformat()
		for composition in compositions:
			record = self._normalize_composition(composition)
			cursor.execute("""
				INSERT OR REPLACE INTO offline_compositions
				(composition_id, name, description, type, capability_count,
				 validation_score, is_offline_ready, data_json, last_sync)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
			""", (
				record["composition_id"], record["name"], record["description"],
				record["type"], record["capability_count"], record["validation_score"],
				record["is_offline_ready"], json.dumps(record["data"], default=str),
				synced_at
			))

		conn.commit()
		conn.close()

		return len(compositions)

	def _normalize_capability(self, capability: Any) -> Dict[str, Any]:
		"""Normalize online capability records into offline capability rows."""
		capability_id = self._field(capability, "capability_id", "id")
		code = self._field(capability, "capability_code", "code", default=capability_id)
		name = self._field(capability, "capability_name", "name", default=code)
		status = self._field(capability, "status", default="active")

		return {
			"capability_id": capability_id,
			"code": code,
			"name": name,
			"description": self._field(capability, "description", default="") or "",
			"category": self._field(capability, "category", default="uncategorized") or "uncategorized",
			"version": self._field(capability, "version", default="1.0.0") or "1.0.0",
			"quality_score": float(self._field(capability, "quality_score", default=0.0) or 0.0),
			"status": self._value(status),
			"data": self._record_to_json(capability)
		}

	def _normalize_composition(self, composition: Any) -> Dict[str, Any]:
		"""Normalize online composition records into offline composition rows."""
		composition_id = self._field(composition, "composition_id", "id")
		capabilities = self._field(composition, "capability_ids", "capabilities", default=[]) or []
		validation_results = self._field(composition, "validation_results", default={}) or {}
		validation_status = self._value(self._field(composition, "validation_status", "status", default="valid"))
		validation_score = self._field(composition, "validation_score", default=None)
		if validation_score is None and isinstance(validation_results, dict):
			validation_score = validation_results.get("validation_score", 0.0)

		return {
			"composition_id": composition_id,
			"name": self._field(composition, "name", default=composition_id),
			"description": self._field(composition, "description", default="") or "",
			"type": self._value(self._field(composition, "composition_type", "type", default="custom")),
			"capability_count": int(self._field(composition, "capability_count", default=len(capabilities)) or 0),
			"validation_score": float(validation_score or 0.0),
			"is_offline_ready": validation_status in {"valid", "warning", "active"},
			"data": self._record_to_json(composition)
		}

	def _record_updated_since(self, record: Any, since: Optional[datetime]) -> bool:
		"""Return whether an online record should be included in this sync."""
		if since is None:
			return True
		updated_at = self._field(record, "updated_at", "last_sync", "created_at", default=None)
		if not updated_at:
			return True
		if isinstance(updated_at, str):
			try:
				updated_at = datetime.fromisoformat(updated_at)
			except ValueError:
				return True
		return updated_at >= since

	def _record_to_json(self, record: Any) -> Dict[str, Any]:
		"""Convert online service records to JSON-safe dictionaries."""
		if isinstance(record, dict):
			return dict(record)
		return {
			key: self._value(value)
			for key, value in vars(record).items()
			if not key.startswith("_")
		}

	def _field(self, record: Any, *names: str, default: Any = None) -> Any:
		"""Read a field from a dict or object using fallback names."""
		for name in names:
			if isinstance(record, dict) and name in record:
				return record[name]
			if not isinstance(record, dict) and hasattr(record, name):
				return getattr(record, name)
		return default

	def _value(self, value: Any) -> Any:
		"""Return primitive values for enums and datetime objects."""
		if hasattr(value, "value"):
			return value.value
		if isinstance(value, datetime):
			return value.isoformat()
		return value

	async def _get_sync_manifest(self) -> Optional[SyncManifest]:
		"""Get current sync manifest."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		cursor.execute("""
			SELECT tenant_id, last_sync, capabilities_count, compositions_count,
				   offline_actions_count, sync_version, checksum
			FROM sync_manifest
			WHERE tenant_id = ?
		""", (self.tenant_id,))

		row = cursor.fetchone()
		conn.close()

		if row:
			return SyncManifest(
				tenant_id=row[0],
				last_sync=datetime.fromisoformat(row[1]) if row[1] else datetime.utcnow(),
				capabilities_count=row[2],
				compositions_count=row[3],
				offline_actions_count=row[4],
				sync_version=row[5],
				checksum=row[6]
			)
		return None

	async def _update_sync_manifest(
		self,
		sync_time: datetime,
		capabilities_count: int,
		compositions_count: int
	):
		"""Update sync manifest."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		# Get offline actions count
		cursor.execute("SELECT COUNT(*) FROM offline_actions WHERE status = 'pending'")
		actions_count = cursor.fetchone()[0]

		# Generate checksum (simplified)
		checksum = f"{sync_time.isoformat()}_{capabilities_count}_{compositions_count}"

		cursor.execute("""
			INSERT OR REPLACE INTO sync_manifest
			(tenant_id, last_sync, capabilities_count, compositions_count,
			 offline_actions_count, sync_version, checksum)
			VALUES (?, ?, ?, ?, ?, ?, ?)
		""", (
			self.tenant_id, sync_time.isoformat(), capabilities_count,
			compositions_count, actions_count, "1.0", checksum
		))

		conn.commit()
		conn.close()

	# =========================================================================
	# Mobile UI Optimization
	# =========================================================================

	async def get_mobile_dashboard_data(self) -> Dict[str, Any]:
		"""Get dashboard data optimized for mobile."""
		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		# Get counts
		cursor.execute("SELECT COUNT(*) FROM offline_capabilities")
		capabilities_count = cursor.fetchone()[0]

		cursor.execute("SELECT COUNT(*) FROM offline_compositions")
		compositions_count = cursor.fetchone()[0]

		cursor.execute("SELECT COUNT(*) FROM offline_actions WHERE status = 'pending'")
		pending_actions = cursor.fetchone()[0]

		# Get recent capabilities
		cursor.execute("""
			SELECT name, category, quality_score
			FROM offline_capabilities
			ORDER BY last_sync DESC LIMIT 5
		""")
		recent_capabilities = [
			{"name": row[0], "category": row[1], "quality_score": row[2]}
			for row in cursor.fetchall()
		]

		conn.close()

		return {
			"capabilities_count": capabilities_count,
			"compositions_count": compositions_count,
			"pending_sync_actions": pending_actions,
			"is_online": self.is_online,
			"last_sync": self.last_sync.isoformat() if self.last_sync else None,
			"recent_capabilities": recent_capabilities,
			"storage_info": {
				"db_size_mb": Path(self.offline_db_path).stat().st_size / (1024 * 1024),
				"available_space": "unlimited"  # Simplified
			}
		}

	async def cleanup_offline_data(self, older_than_days: int = 30) -> Dict[str, Any]:
		"""Clean up old offline data."""
		cutoff_date = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()

		conn = sqlite3.connect(self.offline_db_path)
		cursor = conn.cursor()

		# Clean up old completed actions
		cursor.execute("""
			DELETE FROM offline_actions
			WHERE status = 'completed' AND created_at < ?
		""", (cutoff_date,))
		actions_cleaned = cursor.rowcount

		# Clean up old failed actions
		cursor.execute("""
			DELETE FROM offline_actions
			WHERE status = 'failed' AND created_at < ?
		""", (cutoff_date,))
		failed_cleaned = cursor.rowcount

		conn.commit()

		# Vacuum database
		cursor.execute("VACUUM")
		conn.close()

		return {
			"actions_cleaned": actions_cleaned,
			"failed_actions_cleaned": failed_cleaned,
			"cleanup_date": cutoff_date
		}

# =============================================================================
# Progressive Web App Support
# =============================================================================

class PWAManifest(BaseModel):
	"""Progressive Web App manifest model."""
	model_config = ConfigDict(extra='forbid')

	name: str = "APG Capability Registry"
	short_name: str = "APG Registry"
	description: str = "APG Capability Registry Mobile App"
	start_url: str = "/"
	display: str = "standalone"
	background_color: str = "#ffffff"
	theme_color: str = "#2563eb"
	orientation: str = "portrait"
	icons: List[Dict[str, Any]] = Field(default_factory=list)

def generate_pwa_manifest() -> PWAManifest:
	"""Generate PWA manifest."""
	return PWAManifest(
		icons=[
			{
				"src": "/static/icons/icon-192x192.png",
				"sizes": "192x192",
				"type": "image/png"
			},
			{
				"src": "/static/icons/icon-512x512.png",
				"sizes": "512x512",
				"type": "image/png"
			}
		]
	)

def generate_service_worker() -> str:
	"""Generate service worker JavaScript."""
	return """
// APG Capability Registry Service Worker
const CACHE_NAME = 'apg-registry-v1';
const urlsToCache = [
	'/',
	'/static/css/mobile.css',
	'/static/js/mobile.js',
	'/static/icons/icon-192x192.png',
	'/static/icons/icon-512x512.png'
];

self.addEventListener('install', function(event) {
	event.waitUntil(
		caches.open(CACHE_NAME)
			.then(function(cache) {
				return cache.addAll(urlsToCache);
			})
	);
});

self.addEventListener('fetch', function(event) {
	event.respondWith(
		caches.match(event.request)
			.then(function(response) {
				// Return cached version or fetch from network
				return response || fetch(event.request);
			}
		)
	);
});

// Background sync for offline actions
self.addEventListener('sync', function(event) {
	if (event.tag === 'background-sync') {
		event.waitUntil(syncOfflineActions());
	}
});

async function syncOfflineActions() {
	try {
		const response = await fetch('/api/mobile/sync-offline-actions', {
			method: 'POST'
		});
		return response;
	} catch (error) {
		console.log('Background sync failed:', error);
		throw error;
	}
}
"""

# Export service
__all__ = [
	"MobileOfflineService",
	"MobileCapabilityView",
	"MobileCompositionView",
	"OfflineAction",
	"SyncManifest",
	"PWAManifest",
	"generate_pwa_manifest",
	"generate_service_worker"
]
