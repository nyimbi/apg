"""Compatibility facade for composition deployment automation imports."""

from enum import Enum

from .deployment import (
	DeploymentAutomationService,
	DeploymentConfig,
	DeploymentEnvironment,
	DeploymentResult,
	DeploymentStatus,
	DeploymentTarget,
	get_deployment_service,
)


class DeploymentStrategy(str, Enum):
	ROLLING_UPDATE = "rolling_update"
	BLUE_GREEN = "blue_green"
	CANARY = "canary"
	RECREATE = "recreate"
	A_B_TESTING = "a_b_testing"
	MICROSERVICES = "microservices"
	HYBRID_CLOUD = "hybrid_cloud"
	EDGE_DISTRIBUTED = "edge_distributed"
	SERVERLESS = "serverless"

__all__ = [
	"DeploymentAutomationService",
	"DeploymentStrategy",
	"DeploymentEnvironment",
	"DeploymentStatus",
	"DeploymentTarget",
	"DeploymentConfig",
	"DeploymentResult",
	"get_deployment_service",
]
