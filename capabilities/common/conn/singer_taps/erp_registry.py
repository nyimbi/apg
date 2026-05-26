"""
ERP Connector Registry
Central registry for all Enterprise Resource Planning system connectors

This module provides a unified interface for discovering and configuring
all available ERP connectors in the APG Connection Management system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ERPSystemType(Enum):
    """Supported ERP system types"""
    # SAP Ecosystem
    SAP_ERP = "sap_erp"
    SAP_S4HANA = "sap_s4hana"
    SAP_BUSINESS_ONE = "sap_business_one"
    SAP_SUCCESSFACTORS = "sap_successfactors"
    SAP_CONCUR = "sap_concur"
    SAP_ARIBA = "sap_ariba"
    SAP_FIELDGLASS = "sap_fieldglass"

    # Microsoft Dynamics
    DYNAMICS_365_FO = "dynamics_365_finance_operations"
    DYNAMICS_365_BC = "dynamics_365_business_central"
    DYNAMICS_365_SALES = "dynamics_365_sales"
    DYNAMICS_365_SERVICE = "dynamics_365_customer_service"
    DYNAMICS_365_MARKETING = "dynamics_365_marketing"
    DYNAMICS_365_SCM = "dynamics_365_supply_chain"
    DYNAMICS_AX = "dynamics_ax"
    DYNAMICS_NAV = "dynamics_nav"

    # Oracle Systems
    ORACLE_CLOUD_ERP = "oracle_cloud_erp"
    ORACLE_FUSION = "oracle_fusion"
    ORACLE_EBS = "oracle_ebs"
    ORACLE_JDE = "oracle_jd_edwards"
    ORACLE_PEOPLESOFT = "oracle_peoplesoft"

    # NetSuite
    NETSUITE_ERP = "netsuite_erp"
    NETSUITE_CRM = "netsuite_crm"
    NETSUITE_ECOMMERCE = "netsuite_ecommerce"

    # Workday
    WORKDAY_HCM = "workday_hcm"
    WORKDAY_FINANCIAL = "workday_financial"
    WORKDAY_PLANNING = "workday_planning"

    # Sage Systems
    SAGE_X3 = "sage_x3"
    SAGE_100 = "sage_100"
    SAGE_300 = "sage_300"
    SAGE_INTACCT = "sage_intacct"
    SAGE_PEOPLE = "sage_people"


@dataclass
class ERPConnectorInfo:
    """Information about an ERP connector"""
    system_type: ERPSystemType
    display_name: str
    vendor: str
    description: str
    tap_class: str
    client_class: str
    supported_versions: List[str]
    authentication_methods: List[str]
    data_categories: List[str]
    stream_count: int
    implementation_status: str  # "complete", "beta", "planned"
    documentation_url: Optional[str] = None
    configuration_template: Optional[Dict] = None


class ERPConnectorRegistry:
    """Central registry for all ERP connectors"""

    def __init__(self):
        self._connectors = {}
        self._initialize_registry()

    def _initialize_registry(self) -> None:
        """Initialize the connector registry with all available ERP systems"""

        # SAP Ecosystem
        self._register_sap_connectors()

        # Microsoft Dynamics
        self._register_dynamics_connectors()

        # Oracle Systems
        self._register_oracle_connectors()

        # NetSuite
        self._register_netsuite_connectors()

        # Workday
        self._register_workday_connectors()

        # Sage Systems
        self._register_sage_connectors()

    def _register_sap_connectors(self) -> None:
        """Register SAP ecosystem connectors"""

        # SAP ERP/S4HANA
        self._connectors[ERPSystemType.SAP_ERP] = ERPConnectorInfo(
            system_type=ERPSystemType.SAP_ERP,
            display_name="SAP ERP (ECC)",
            vendor="SAP",
            description="Traditional SAP R/3 and ERP Central Component with comprehensive business functionality",
            tap_class="tap_sap.TapSAP",
            client_class="tap_sap.SAPClient",
            supported_versions=["ECC 6.0", "EHP 7", "EHP 8"],
            authentication_methods=["RFC", "OData", "Basic Auth"],
            data_categories=["Financial", "Supply Chain", "HR", "Manufacturing", "Sales"],
            stream_count=60,
            implementation_status="complete",
            configuration_template={
                "sap_system_type": "erp",
                "host": "sap-server.company.com",
                "client": "100",
                "system_number": "00",
                "username": "integration_user",
                "password": "secure_password",
                "language": "EN"
            }
        )

        self._connectors[ERPSystemType.SAP_S4HANA] = ERPConnectorInfo(
            system_type=ERPSystemType.SAP_S4HANA,
            display_name="SAP S/4HANA",
            vendor="SAP",
            description="Next-generation intelligent ERP suite with real-time analytics and machine learning",
            tap_class="tap_sap.TapSAP",
            client_class="tap_sap.SAPClient",
            supported_versions=["1809", "1909", "2020", "2021", "2022"],
            authentication_methods=["RFC", "OData", "OAuth 2.0"],
            data_categories=["Financial", "Supply Chain", "HR", "Manufacturing", "Sales", "Analytics"],
            stream_count=65,
            implementation_status="complete"
        )

        # SAP Business One
        self._connectors[ERPSystemType.SAP_BUSINESS_ONE] = ERPConnectorInfo(
            system_type=ERPSystemType.SAP_BUSINESS_ONE,
            display_name="SAP Business One",
            vendor="SAP",
            description="Comprehensive ERP solution designed for small and medium enterprises",
            tap_class="tap_sap.TapSAP",
            client_class="tap_sap.SAPClient",
            supported_versions=["9.3", "10.0"],
            authentication_methods=["Service Layer API", "DI API"],
            data_categories=["Financial", "Sales", "Purchasing", "Inventory", "CRM"],
            stream_count=25,
            implementation_status="complete"
        )

    def _register_dynamics_connectors(self) -> None:
        """Register Microsoft Dynamics connectors"""

        # Dynamics 365 Finance & Operations
        self._connectors[ERPSystemType.DYNAMICS_365_FO] = ERPConnectorInfo(
            system_type=ERPSystemType.DYNAMICS_365_FO,
            display_name="Dynamics 365 Finance & Operations",
            vendor="Microsoft",
            description="Comprehensive financial and operational ERP with global capabilities",
            tap_class="tap_dynamics.TapDynamics",
            client_class="tap_dynamics.DynamicsClient",
            supported_versions=["10.0.x"],
            authentication_methods=["Azure AD OAuth 2.0", "Service Principal"],
            data_categories=["Financial", "Supply Chain", "Manufacturing", "Retail", "HR"],
            stream_count=50,
            implementation_status="complete",
            configuration_template={
                "dynamics_system_type": "finance_operations",
                "tenant_id": "your-tenant-id",
                "client_id": "your-client-id",
                "client_secret": "your-client-secret",
                "base_url": "https://your-instance.operations.dynamics.com"
            }
        )

        # Dynamics 365 Business Central
        self._connectors[ERPSystemType.DYNAMICS_365_BC] = ERPConnectorInfo(
            system_type=ERPSystemType.DYNAMICS_365_BC,
            display_name="Dynamics 365 Business Central",
            vendor="Microsoft",
            description="All-in-one business management solution for small to medium businesses",
            tap_class="tap_dynamics.TapDynamics",
            client_class="tap_dynamics.DynamicsClient",
            supported_versions=["Wave 1 2023", "Wave 2 2023", "Wave 1 2024"],
            authentication_methods=["Azure AD OAuth 2.0", "API Keys"],
            data_categories=["Financial", "Sales", "Service", "Operations"],
            stream_count=40,
            implementation_status="complete"
        )

    def _register_oracle_connectors(self) -> None:
        """Register Oracle ERP connectors"""

        # Oracle Cloud ERP
        self._connectors[ERPSystemType.ORACLE_CLOUD_ERP] = ERPConnectorInfo(
            system_type=ERPSystemType.ORACLE_CLOUD_ERP,
            display_name="Oracle Cloud ERP",
            vendor="Oracle",
            description="Complete cloud-based ERP suite with AI-powered insights",
            tap_class="tap_oracle.TapOracle",
            client_class="tap_oracle.OracleClient",
            supported_versions=["23A", "23B", "23C"],
            authentication_methods=["OAuth 2.0", "JWT Token"],
            data_categories=["Financial", "Procurement", "Project Management", "Risk Management"],
            stream_count=45,
            implementation_status="beta",
            configuration_template={
                "oracle_system_type": "cloud_erp",
                "host": "your-instance.oraclecloud.com",
                "username": "integration_user",
                "password": "secure_password",
                "pod": "your_pod_name"
            }
        )

    def _register_netsuite_connectors(self) -> None:
        """Register NetSuite connectors"""

        # NetSuite ERP
        self._connectors[ERPSystemType.NETSUITE_ERP] = ERPConnectorInfo(
            system_type=ERPSystemType.NETSUITE_ERP,
            display_name="NetSuite ERP",
            vendor="Oracle NetSuite",
            description="Cloud-based business management suite with integrated ERP, CRM, and ecommerce",
            tap_class="tap_netsuite.TapNetSuite",
            client_class="tap_netsuite.NetSuiteClient",
            supported_versions=["2023.1", "2023.2", "2024.1"],
            authentication_methods=["Token Based Authentication", "OAuth 2.0"],
            data_categories=["Financial", "CRM", "Inventory", "Ecommerce", "Analytics"],
            stream_count=35,
            implementation_status="beta",
            configuration_template={
                "account_id": "your_account_id",
                "consumer_key": "your_consumer_key",
                "consumer_secret": "your_consumer_secret",
                "token_id": "your_token_id",
                "token_secret": "your_token_secret"
            }
        )

    def _register_workday_connectors(self) -> None:
        """Register Workday connectors"""

        # Workday HCM
        self._connectors[ERPSystemType.WORKDAY_HCM] = ERPConnectorInfo(
            system_type=ERPSystemType.WORKDAY_HCM,
            display_name="Workday HCM",
            vendor="Workday",
            description="Cloud-based Human Capital Management platform with advanced analytics",
            tap_class="tap_workday.TapWorkday",
            client_class="tap_workday.WorkdayClient",
            supported_versions=["2023R1", "2023R2", "2024R1"],
            authentication_methods=["OAuth 2.0", "Username/Password"],
            data_categories=["HR", "Payroll", "Benefits", "Talent Management", "Analytics"],
            stream_count=30,
            implementation_status="beta",
            configuration_template={
                "workday_system_type": "hcm",
                "tenant": "your_tenant",
                "base_url": "https://services1.myworkday.com",
                "username": "integration_user",
                "password": "secure_password"
            }
        )

    def _register_sage_connectors(self) -> None:
        """Register Sage ERP connectors"""

        # Sage X3
        self._connectors[ERPSystemType.SAGE_X3] = ERPConnectorInfo(
            system_type=ERPSystemType.SAGE_X3,
            display_name="Sage X3",
            vendor="Sage",
            description="Mid-market ERP solution for manufacturing and distribution companies",
            tap_class="tap_sage.TapSage",
            client_class="tap_sage.SageClient",
            supported_versions=["Version 12", "Version 11"],
            authentication_methods=["Web Services", "Database Direct"],
            data_categories=["Financial", "Manufacturing", "Distribution", "CRM"],
            stream_count=25,
            implementation_status="planned"
        )

    def get_connector(self, system_type: ERPSystemType) -> Optional[ERPConnectorInfo]:
        """Get connector information for a specific ERP system type"""
        return self._connectors.get(system_type)

    def list_connectors(self,
                       vendor: Optional[str] = None,
                       status: Optional[str] = None,
                       category: Optional[str] = None) -> List[ERPConnectorInfo]:
        """List available connectors with optional filtering"""
        connectors = list(self._connectors.values())

        if vendor:
            connectors = [c for c in connectors if c.vendor.lower() == vendor.lower()]

        if status:
            connectors = [c for c in connectors if c.implementation_status == status]

        if category:
            connectors = [c for c in connectors if category in c.data_categories]

        return sorted(connectors, key=lambda x: x.display_name)

    def get_vendors(self) -> List[str]:
        """Get list of all ERP vendors"""
        vendors = set(c.vendor for c in self._connectors.values())
        return sorted(list(vendors))

    def get_implementation_status_summary(self) -> Dict[str, int]:
        """Get summary of implementation status across all connectors"""
        status_counts = {}
        for connector in self._connectors.values():
            status = connector.implementation_status
            status_counts[status] = status_counts.get(status, 0) + 1

        return status_counts

    def get_stream_count_total(self) -> int:
        """Get total number of streams across all connectors"""
        return sum(c.stream_count for c in self._connectors.values())

    def get_configuration_template(self, system_type: ERPSystemType) -> Optional[Dict]:
        """Get configuration template for a specific ERP system"""
        connector = self._connectors.get(system_type)
        return connector.configuration_template if connector else None

    def validate_configuration(self, system_type: ERPSystemType, config: Dict) -> List[str]:
        """Validate configuration for a specific ERP system"""
        connector = self.get_connector(system_type)
        if not connector:
            return [f"Unknown ERP system type: {system_type}"]

        errors = []
        template = connector.configuration_template

        if template:
            # Check required fields
            for required_field in template.keys():
                if required_field not in config:
                    errors.append(f"Missing required field: {required_field}")

        return errors


# Global registry instance
erp_registry = ERPConnectorRegistry()


def get_erp_registry() -> ERPConnectorRegistry:
    """Get the global ERP connector registry"""
    return erp_registry


def list_supported_erp_systems() -> List[str]:
    """Get list of all supported ERP system names"""
    return [connector.display_name for connector in erp_registry.list_connectors()]


def get_erp_connector_info(system_name: str) -> Optional[ERPConnectorInfo]:
    """Get connector info by display name"""
    for connector in erp_registry.list_connectors():
        if connector.display_name.lower() == system_name.lower():
            return connector
    return None