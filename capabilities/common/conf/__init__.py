"""APG Configuration Management capability.

Standalone package: ``pip install apg-common-conf``

Quick start::

    from apg_common_conf import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : conf
Provides      : conf_operations, conf_agents, review_evidence
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-conf"
__capability_id__ = "conf"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

from .service import (  # noqa: E402
    ProductionConfigurationManager,
    create_configuration_manager,
    get_config_manager,
)

# Alias used by tests
RevolutionaryConfigurationManager = ProductionConfigurationManager

from .models import (  # noqa: E402
    CMResource,
    CMTemplate,
    CMPolicy,
    CMEnvironment,
    CMDeployment,
    ResourceState,
    DeploymentStatus,
    ResourceType,
    CloudProvider,
)

__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
    "ProductionConfigurationManager",
    "RevolutionaryConfigurationManager",
    "create_configuration_manager",
    "get_config_manager",
    "CMResource",
    "CMTemplate",
    "CMPolicy",
    "CMEnvironment",
    "CMDeployment",
    "ResourceState",
    "DeploymentStatus",
    "ResourceType",
    "CloudProvider",
]
