# Author: Nyimbi Odero
# Company: Datacraft
# Copyright: © 2025

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class DataFormat(str, Enum):
	CSV = "csv"
	PARQUET = "parquet"
	JSON = "json"
	AVRO = "avro"
	ORC = "orc"
	DELTA = "delta"
	ICEBERG = "iceberg"
	OTHER = "other"


class DatasetStatus(str, Enum):
	ACTIVE = "active"
	DEPRECATED = "deprecated"
	ARCHIVED = "archived"
	DRAFT = "draft"


class DatasetTag(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	key: str = Field(..., min_length=1, max_length=128, description="Tag key")
	value: str = Field(..., min_length=1, max_length=512, description="Tag value")
	tenant_id: str = Field(..., description="Owning tenant")

	@field_validator("key")
	@classmethod
	def key_no_spaces(cls, v: str) -> str:
		if " " in v:
			raise ValueError("Tag key must not contain spaces")
		return v.lower()


class DataQualityDimension(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str = Field(..., description="Dimension name e.g. completeness, accuracy")
	score: float = Field(..., ge=0.0, le=1.0, description="Score 0–1")
	details: str | None = Field(None, description="Optional explanation")


class QualityScore(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	dataset_id: str = Field(..., description="Associated dataset")
	tenant_id: str = Field(..., description="Owning tenant")
	overall: float = Field(..., ge=0.0, le=1.0, description="Aggregate quality score")
	dimensions: list[DataQualityDimension] = Field(default_factory=list)
	computed_at: datetime = Field(default_factory=datetime.utcnow)
	notes: str | None = None


class Dataset(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Owning tenant")
	name: str = Field(..., min_length=1, max_length=256, description="Human-readable name")
	qualified_name: str = Field(..., description="Fully-qualified unique name, e.g. db.schema.table")
	description: str | None = None
	format: DataFormat = Field(DataFormat.OTHER, description="Storage format")
	status: DatasetStatus = Field(DatasetStatus.ACTIVE)
	location: str | None = Field(None, description="URI / path to the dataset")
	owner: str | None = Field(None, description="Team or user owning the dataset")
	schema_def: dict[str, Any] | None = Field(None, description="JSON schema or column map")
	tags: list[DatasetTag] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

	# Apache Atlas compatibility fields
	type_name: str = Field("DataSet", description="Atlas typeName")
	guid: str | None = Field(None, description="Atlas GUID — set on Atlas registration")
	classifications: list[str] = Field(default_factory=list, description="Atlas classifications")
	business_metadata: dict[str, Any] = Field(default_factory=dict, description="Atlas businessMetadata")


class LineageEdgeType(str, Enum):
	DERIVED_FROM = "DERIVED_FROM"
	COPIES = "COPIES"
	TRANSFORMS = "TRANSFORMS"
	JOINS = "JOINS"
	FILTERS = "FILTERS"
	AGGREGATES = "AGGREGATES"


class LineageEdge(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str = Field(..., description="Owning tenant")
	source_id: str = Field(..., description="Upstream dataset ID")
	target_id: str = Field(..., description="Downstream dataset ID")
	edge_type: LineageEdgeType = Field(LineageEdgeType.DERIVED_FROM)
	process_name: str | None = Field(None, description="ETL job / query that created this edge")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetSearch(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	tenant_id: str = Field(..., description="Tenant scope for search")
	query: str | None = Field(None, description="Free-text search across name, description, tags")
	format: DataFormat | None = None
	status: DatasetStatus | None = None
	owner: str | None = None
	tag_key: str | None = Field(None, description="Filter by tag key")
	tag_value: str | None = Field(None, description="Filter by tag value (requires tag_key)")
	classification: str | None = Field(None, description="Atlas classification filter")
	limit: int = Field(50, ge=1, le=500)
	offset: int = Field(0, ge=0)
