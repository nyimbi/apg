"""Compatibility facade for composition capability-registry imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CRCapabilityStatus(str, Enum):
	ACTIVE = "active"
	INACTIVE = "inactive"
	DEPRECATED = "deprecated"


class CRDependencyType(str, Enum):
	REQUIRED = "required"
	OPTIONAL = "optional"
	CONFLICTS = "conflicts"


class CRCompositionType(str, Enum):
	ENTERPRISE = "enterprise"
	DEPARTMENTAL = "departmental"
	MICROSERVICE = "microservice"
	HYBRID = "hybrid"


class CRVersionConstraint(str, Enum):
	EXACT = "exact"
	MINIMUM = "minimum"
	COMPATIBLE = "compatible"


class ConflictSeverity(str, Enum):
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class RecommendationType(str, Enum):
	ADD = "add"
	REMOVE = "remove"
	REPLACE = "replace"
	CONFIGURE = "configure"


@dataclass
class CRCapability:
	id: str
	name: str = ""
	category: str = ""
	status: CRCapabilityStatus = CRCapabilityStatus.ACTIVE


@dataclass
class CRDependency:
	source_id: str = ""
	target_id: str = ""
	dependency_type: CRDependencyType = CRDependencyType.REQUIRED


@dataclass
class CRComposition:
	id: str
	capability_ids: List[str] = field(default_factory=list)
	composition_type: CRCompositionType = CRCompositionType.ENTERPRISE


@dataclass
class CRVersion:
	version: str = "1.0.0"


@dataclass
class CRMetadata:
	values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRRegistry:
	capabilities: Dict[str, CRCapability] = field(default_factory=dict)


@dataclass
class ConflictReport:
	message: str = ""
	severity: ConflictSeverity = ConflictSeverity.LOW


@dataclass
class CompositionRecommendation:
	message: str = ""
	recommendation_type: RecommendationType = RecommendationType.CONFIGURE


@dataclass
class PerformanceImpact:
	latency_ms: float = 0.0
	memory_mb: float = 0.0


@dataclass
class CompositionValidationResult:
	valid: bool = True
	conflicts: List[ConflictReport] = field(default_factory=list)
	recommendations: List[CompositionRecommendation] = field(default_factory=list)
	performance_impact: Optional[PerformanceImpact] = None


@dataclass
class APGTenantContext:
	tenant_id: str
	user_id: str = "system"
	user_roles: List[str] = field(default_factory=list)
	permissions: List[str] = field(default_factory=list)


@dataclass
class CRServiceResponse:
	success: bool
	data: Any = None
	error: Optional[str] = None


CAPABILITY_METADATA = {
	"capability_id": "composition.capability_registry",
	"name": "Capability Registry",
	"version": "1.0.0",
	"description": "Compatibility registry facade for composition imports",
}


class CRService:
	async def get_db_session(self, tenant_id: str = "default") -> None:
		return None

	async def discover_capabilities(self, tenant_id: str = "default") -> List[CRCapability]:
		return []


class APGIntegrationService:
	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id


class IntelligentCompositionEngine:
	def __init__(self, db_session: Any = None, tenant_id: str = "default", user_id: str = "system") -> None:
		self.db_session = db_session
		self.tenant_id = tenant_id
		self.user_id = user_id

	async def validate_composition(self, **kwargs: Any) -> CompositionValidationResult:
		return CompositionValidationResult(valid=True)

	async def generate_business_recommendations(self, **kwargs: Any) -> List[CompositionRecommendation]:
		return []


_registry_service = CRService()
_apg_services: Dict[str, APGIntegrationService] = {}


def get_registry_service() -> CRService:
	return _registry_service


def get_apg_integration_service(tenant_id: str = "default") -> APGIntegrationService:
	if tenant_id not in _apg_services:
		_apg_services[tenant_id] = APGIntegrationService(tenant_id)
	return _apg_services[tenant_id]


def get_composition_engine(
	db_session: Any = None,
	tenant_id: str = "default",
	user_id: str = "system",
) -> IntelligentCompositionEngine:
	return IntelligentCompositionEngine(db_session=db_session, tenant_id=tenant_id, user_id=user_id)
