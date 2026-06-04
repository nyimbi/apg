"""Pydantic v2 models for APG Data Warehouse (bia_dwh)."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7

def uuid7str() -> str:
	return str(uuid7())

class SchemaType(str, Enum):
	STAR = "star"; SNOWFLAKE = "snowflake"; GALAXY = "galaxy"; FLAT = "flat"; DATA_VAULT = "data_vault"

class TableType(str, Enum):
	FACT = "fact"; DIMENSION = "dimension"; BRIDGE = "bridge"; AGGREGATE = "aggregate"
	STAGING = "staging"; RAW = "raw"; QUARANTINE = "quarantine"

class ETLState(str, Enum):
	PENDING = "pending"; RUNNING = "running"; COMPLETED = "completed"
	FAILED = "failed"; CANCELLED = "cancelled"; RETRYING = "retrying"

class LoadStrategy(str, Enum):
	FULL_REFRESH = "full_refresh"; INCREMENTAL = "incremental"
	SCD_TYPE1 = "scd_type1"; SCD_TYPE2 = "scd_type2"; SCD_TYPE3 = "scd_type3"
	MERGE = "merge"; APPEND = "append"

class StorageTier(str, Enum):
	HOT = "hot"; WARM = "warm"; COLD = "cold"; ARCHIVE = "archive"

class PartitionStrategy(str, Enum):
	RANGE = "range"; LIST = "list"; HASH = "hash"; COMPOSITE = "composite"; NONE = "none"

_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

class SchemaCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; schema_type: SchemaType; grain: str; owner_id: str
	description: str | None = None; tags: list[str] = Field(default_factory=list)

class SchemaResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	schema_type: SchemaType; grain: str; owner_id: str
	description: str | None = None; tags: list[str] = Field(default_factory=list)
	table_count: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class TableCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; schema_id: str; name: str; table_type: TableType
	columns: list[dict[str, Any]]; owner_id: str
	partition_strategy: PartitionStrategy = PartitionStrategy.NONE
	storage_tier: StorageTier = StorageTier.HOT
	lineage_ref: str | None = None; description: str | None = None

class TableResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; schema_id: str
	name: str; table_type: TableType; columns: list[dict[str, Any]]
	owner_id: str; partition_strategy: PartitionStrategy; storage_tier: StorageTier
	lineage_ref: str | None = None; description: str | None = None
	row_count: int | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class ETLJobCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; source_ref: str; target_table_id: str
	load_strategy: LoadStrategy; owner_id: str
	transform_sql: str | None = None; schedule_cron: str | None = None
	description: str | None = None

class ETLJobResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	source_ref: str; target_table_id: str; load_strategy: LoadStrategy
	owner_id: str; state: ETLState = ETLState.PENDING
	transform_sql: str | None = None; schedule_cron: str | None = None
	last_run_at: datetime | None = None; last_run_rows: int | None = None
	description: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class QualityRuleCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; table_id: str; name: str; rule_type: str
	column: str | None = None; config: dict[str, Any] = Field(default_factory=dict)
	owner_id: str

class QualityRuleResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; table_id: str
	name: str; rule_type: str; column: str | None = None
	config: dict[str, Any] = Field(default_factory=dict); owner_id: str
	last_checked_at: datetime | None = None; last_result: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class LineageRecord(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str
	source_table_id: str; target_table_id: str; etl_job_id: str | None = None
	transformation_description: str | None = None
	recorded_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"
