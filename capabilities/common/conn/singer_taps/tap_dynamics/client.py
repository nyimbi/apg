"""
Microsoft Dynamics Client Implementation
Handles connections to various Microsoft Dynamics systems
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime, timezone
import requests
from requests.auth import HTTPBasicAuth
import urllib.parse
import base64

logger = logging.getLogger(__name__)


class DynamicsConnectionError(Exception):
    """Dynamics connection specific errors"""
    pass


class DynamicsClient:
    """Microsoft Dynamics client supporting multiple system types"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_type = config["dynamics_system_type"]
        self.session = None
        self.access_token = None
        self.token_expires_at = None

        # Azure AD configuration
        self.tenant_id = config["tenant_id"]
        self.client_id = config["client_id"]
        self.client_secret = config["client_secret"]
        self.base_url = config["base_url"].rstrip("/")

        # System-specific configuration
        self.api_version = config.get("api_version")
        self.environment_name = config.get("environment_name")
        self.company_id = config.get("company_id")
        self.data_area_id = config.get("data_area_id", "usmf")
        self.batch_size = config.get("batch_size", 1000)
        self.page_size = config.get("page_size", 1000)
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)

        # Initialize system-specific settings
        self._initialize_system_config()

    def _initialize_system_config(self) -> None:
        """Initialize configuration based on Dynamics system type"""
        if self.system_type == "finance_operations":
            self._init_finance_operations_config()
        elif self.system_type == "business_central":
            self._init_business_central_config()
        elif self.system_type in ["sales", "customer_service", "marketing"]:
            self._init_crm_config()
        elif self.system_type == "supply_chain":
            self._init_supply_chain_config()
        elif self.system_type in ["ax", "nav"]:
            self._init_legacy_config()

    def _init_finance_operations_config(self) -> None:
        """Initialize Finance & Operations specific configuration"""
        self.api_version = self.api_version or "v1.0"
        self.api_path = f"/data"
        self.resource_url = f"{self.base_url}/"

        # Azure AD scope for Finance & Operations
        self.scope = f"{self.resource_url}.default"

    def _init_business_central_config(self) -> None:
        """Initialize Business Central specific configuration"""
        self.api_version = self.api_version or "v2.0"

        if self.environment_name:
            self.api_path = f"/api/{self.api_version}/environments/{self.environment_name}/companies"
            if self.company_id:
                self.api_path += f"/{self.company_id}"
        else:
            self.api_path = f"/api/{self.api_version}"

        self.resource_url = f"{self.base_url}/"
        self.scope = f"{self.resource_url}.default"

    def _init_crm_config(self) -> None:
        """Initialize CRM (Sales/Customer Service/Marketing) configuration"""
        self.api_version = self.api_version or "v9.2"
        self.api_path = f"/api/data/{self.api_version}"
        self.resource_url = f"{self.base_url}/"
        self.scope = f"{self.resource_url}.default"

    def _init_supply_chain_config(self) -> None:
        """Initialize Supply Chain Management configuration"""
        self.api_version = self.api_version or "v1.0"
        self.api_path = f"/data"
        self.resource_url = f"{self.base_url}/"
        self.scope = f"{self.resource_url}.default"

    def _init_legacy_config(self) -> None:
        """Initialize legacy systems (AX/NAV) configuration"""
        if self.system_type == "ax":
            self.api_path = f"/Services/AxdEntity"
        else:  # NAV
            self.api_path = f"/OData"

        # Legacy systems might use different authentication
        self.resource_url = f"{self.base_url}/"
        self.scope = f"{self.resource_url}.default"

    def connect(self) -> None:
        """Establish connection to Dynamics system"""
        logger.info(f"Connecting to Dynamics {self.system_type} at {self.base_url}")

        try:
            # Initialize session
            self.session = requests.Session()
            self.session.timeout = self.timeout

            # Get access token
            self._authenticate()

            logger.info("Successfully connected to Dynamics system")

        except Exception as e:
            raise DynamicsConnectionError(f"Failed to connect to Dynamics: {e}")

    def _authenticate(self) -> None:
        """Authenticate with Azure AD and get access token"""
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

        token_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope
        }

        response = requests.post(token_url, data=token_data)
        response.raise_for_status()

        token_info = response.json()
        self.access_token = token_info["access_token"]

        # Calculate expiration time
        expires_in = token_info.get("expires_in", 3600)
        self.token_expires_at = time.time() + expires_in - 300  # 5 min buffer

        # Set authorization header
        self.session.headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0"
        })

        logger.info("Successfully authenticated with Azure AD")

    def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token"""
        if not self.access_token or time.time() >= self.token_expires_at:
            logger.info("Token expired, re-authenticating...")
            self._authenticate()

    def disconnect(self) -> None:
        """Disconnect from Dynamics system"""
        if self.session:
            self.session.close()
            self.session = None

        self.access_token = None
        self.token_expires_at = None

        logger.info("Disconnected from Dynamics system")

    def get_entity_metadata(self, stream_name: str, stream_config: Dict) -> Dict:
        """Get entity metadata for schema discovery"""
        logger.info(f"Getting metadata for entity: {stream_name}")

        self._ensure_authenticated()

        try:
            entity_name = stream_config.get("entity_name", stream_name)

            if self.system_type == "business_central":
                # Business Central uses different metadata endpoint
                url = f"{self.base_url}{self.api_path}/{entity_name}/$metadata"
            elif self.system_type in ["sales", "customer_service", "marketing"]:
                # CRM systems use OData metadata
                url = f"{self.base_url}{self.api_path}/$metadata"
            else:
                # Finance & Operations and others
                url = f"{self.base_url}{self.api_path}/{entity_name}/$metadata"

            response = self._make_request("GET", url)

            # Parse metadata response (simplified)
            return self._parse_metadata_response(response.text, entity_name)

        except Exception as e:
            logger.error(f"Failed to get metadata for {stream_name}: {e}")
            return {"fields": []}

    def _parse_metadata_response(self, metadata_xml: str, entity_name: str) -> Dict:
        """Parse metadata XML response (simplified implementation)"""
        # This is a simplified parser - in production, you'd use proper XML parsing
        fields = []

        # Common field patterns based on system type
        if self.system_type == "business_central":
            # Business Central common fields
            fields = [
                {"name": "id", "type": "Edm.Guid"},
                {"name": "displayName", "type": "Edm.String"},
                {"name": "lastModifiedDateTime", "type": "Edm.DateTimeOffset"},
                {"name": "number", "type": "Edm.String"}
            ]
        elif self.system_type in ["sales", "customer_service", "marketing"]:
            # CRM common fields
            fields = [
                {"name": "id", "type": "Edm.Guid"},
                {"name": "createdon", "type": "Edm.DateTimeOffset"},
                {"name": "modifiedon", "type": "Edm.DateTimeOffset"},
                {"name": "ownerid", "type": "Edm.Guid"},
                {"name": "statecode", "type": "Edm.Int32"},
                {"name": "statuscode", "type": "Edm.Int32"}
            ]
        else:
            # Finance & Operations common fields
            fields = [
                {"name": "dataAreaId", "type": "Edm.String"},
                {"name": "RecId", "type": "Edm.Int64"},
                {"name": "ModifiedDateTime", "type": "Edm.DateTimeOffset"},
                {"name": "CreatedDateTime", "type": "Edm.DateTimeOffset"}
            ]

        return {"fields": fields}

    def get_records(self, stream_name: str, stream_config: Dict, start_date: Optional[str] = None) -> Iterator[Dict]:
        """Get records from Dynamics system"""
        logger.info(f"Getting records for stream: {stream_name}")

        entity_name = stream_config.get("entity_name", stream_name)
        date_field = stream_config.get("date_field", "modifiedon")

        # Build base URL
        if self.system_type == "finance_operations":
            base_url = f"{self.base_url}{self.api_path}/{entity_name}"
        elif self.system_type == "business_central":
            base_url = f"{self.base_url}{self.api_path}/{entity_name}"
        elif self.system_type in ["sales", "customer_service", "marketing"]:
            base_url = f"{self.base_url}{self.api_path}/{entity_name}"
        else:
            base_url = f"{self.base_url}{self.api_path}/{entity_name}"

        # Get data with pagination
        skip = 0
        while True:
            params = {
                "$top": self.page_size,
                "$skip": skip
            }

            # Add date filter for incremental sync
            if start_date and date_field:
                filter_clause = f"{date_field} gt {start_date}"
                params["$filter"] = filter_clause

            # Add system-specific parameters
            if self.system_type == "finance_operations":
                params["$format"] = "json"
                if self.data_area_id:
                    if "$filter" in params:
                        params["$filter"] += f" and dataAreaId eq '{self.data_area_id}'"
                    else:
                        params["$filter"] = f"dataAreaId eq '{self.data_area_id}'"

            try:
                response = self._make_request("GET", base_url, params=params)
                data = response.json()

                # Extract records based on system type
                if self.system_type == "business_central":
                    records = data.get("value", [])
                elif self.system_type in ["sales", "customer_service", "marketing"]:
                    records = data.get("value", [])
                elif self.system_type == "finance_operations":
                    records = data.get("value", [])
                else:
                    records = data if isinstance(data, list) else [data]

                if not records:
                    break

                for record in records:
                    yield record

                # Check if we have more pages
                if len(records) < self.page_size:
                    break

                skip += self.page_size

            except Exception as e:
                logger.error(f"Error fetching records: {e}")
                break

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic"""
        self._ensure_authenticated()

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)

                if response.status_code == 401:
                    # Token might be expired, try re-authenticating
                    logger.info("Received 401, re-authenticating...")
                    self._authenticate()
                    continue

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    raise

                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)

        raise DynamicsConnectionError(f"Request failed after {self.max_retries + 1} attempts")

    def test_connection(self) -> bool:
        """Test connection to Dynamics system"""
        try:
            self.connect()

            # Test a simple request
            if self.system_type == "business_central":
                test_url = f"{self.base_url}{self.api_path}/companies"
            elif self.system_type in ["sales", "customer_service", "marketing"]:
                test_url = f"{self.base_url}{self.api_path}/organizations"
            elif self.system_type == "finance_operations":
                test_url = f"{self.base_url}{self.api_path}/Companies"
            else:
                test_url = f"{self.base_url}{self.api_path}"

            response = self._make_request("GET", test_url, params={"$top": 1})
            logger.info(f"Connection test successful - Status: {response.status_code}")

            self.disconnect()
            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False