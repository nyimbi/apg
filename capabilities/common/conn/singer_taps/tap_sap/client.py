"""
SAP Client Implementation
Handles connections to various SAP systems and data extraction
"""

import logging
import json
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime, timezone
import requests
from requests.auth import HTTPBasicAuth
import urllib.parse

try:
    import pyrfc
    HAS_RFC = True
except ImportError:
    HAS_RFC = False
    logging.warning("pyrfc not available - RFC connections will not work")

logger = logging.getLogger(__name__)


class SAPConnectionError(Exception):
    """SAP connection specific errors"""
    pass


class SAPClient:
    """SAP client supporting multiple SAP system types"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_type = config["sap_system_type"]
        self.connection = None
        self.session = None

        # Common configuration
        self.host = config["host"]
        self.username = config["username"]
        self.password = config["password"]

        # SAP ERP/S4HANA specific
        self.client = config.get("client", "001")
        self.system_number = config.get("system_number", "00")
        self.language = config.get("language", "EN")
        self.router = config.get("router")

        # API specific
        self.api_version = config.get("api_version")
        self.batch_size = config.get("batch_size", 1000)

        # Initialize system-specific settings
        self._initialize_system_config()

    def _initialize_system_config(self) -> None:
        """Initialize configuration based on SAP system type"""
        if self.system_type in ["erp", "s4hana"]:
            self._init_erp_config()
        elif self.system_type == "business_one":
            self._init_business_one_config()
        elif self.system_type == "successfactors":
            self._init_successfactors_config()
        elif self.system_type == "concur":
            self._init_concur_config()
        elif self.system_type == "ariba":
            self._init_ariba_config()
        elif self.system_type == "fieldglass":
            self._init_fieldglass_config()

    def _init_erp_config(self) -> None:
        """Initialize ERP/S4HANA specific configuration"""
        self.use_rfc = True
        self.base_url = f"http://{self.host}:8000/sap/opu/odata/sap/"

        if not HAS_RFC:
            logger.warning("pyrfc not available - falling back to OData API")
            self.use_rfc = False

    def _init_business_one_config(self) -> None:
        """Initialize Business One specific configuration"""
        self.use_rfc = False
        self.base_url = f"https://{self.host}:50000/b1s/v1/"
        self.login_url = f"{self.base_url}Login"

    def _init_successfactors_config(self) -> None:
        """Initialize SuccessFactors specific configuration"""
        self.use_rfc = False
        self.base_url = f"https://{self.host}/odata/v2/"
        self.api_version = self.api_version or "v2"

    def _init_concur_config(self) -> None:
        """Initialize Concur specific configuration"""
        self.use_rfc = False
        self.base_url = f"https://{self.host}/api/"
        self.api_version = self.api_version or "v3.0"

    def _init_ariba_config(self) -> None:
        """Initialize Ariba specific configuration"""
        self.use_rfc = False
        self.base_url = f"https://{self.host}/api/"
        self.api_version = self.api_version or "v1"

    def _init_fieldglass_config(self) -> None:
        """Initialize Fieldglass specific configuration"""
        self.use_rfc = False
        self.base_url = f"https://{self.host}/api/"
        self.api_version = self.api_version or "v1"

    def connect(self) -> None:
        """Establish connection to SAP system"""
        logger.info(f"Connecting to SAP {self.system_type} system at {self.host}")

        try:
            if self.system_type in ["erp", "s4hana"] and self.use_rfc:
                self._connect_rfc()
            else:
                self._connect_http()

            logger.info("Successfully connected to SAP system")

        except Exception as e:
            raise SAPConnectionError(f"Failed to connect to SAP system: {e}")

    def _connect_rfc(self) -> None:
        """Connect using RFC (for ERP/S4HANA)"""
        if not HAS_RFC:
            raise SAPConnectionError("pyrfc library not available")

        connection_params = {
            'ashost': self.host,
            'sysnr': self.system_number,
            'client': self.client,
            'user': self.username,
            'passwd': self.password,
            'lang': self.language
        }

        if self.router:
            connection_params['saprouter'] = self.router

        self.connection = pyrfc.Connection(**connection_params)
        logger.info("RFC connection established")

    def _connect_http(self) -> None:
        """Connect using HTTP/REST APIs"""
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(self.username, self.password)
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        # System-specific authentication
        if self.system_type == "business_one":
            self._authenticate_business_one()
        elif self.system_type == "successfactors":
            self._authenticate_successfactors()

        logger.info("HTTP connection established")

    def _authenticate_business_one(self) -> None:
        """Authenticate with SAP Business One"""
        login_data = {
            "UserName": self.username,
            "Password": self.password,
            "CompanyDB": self.config.get("company_db", "SBODEMOUS")
        }

        response = self.session.post(self.login_url, json=login_data)
        response.raise_for_status()

        # Session cookie is automatically handled by requests.Session
        logger.info("Business One authentication successful")

    def _authenticate_successfactors(self) -> None:
        """Authenticate with SAP SuccessFactors"""
        # SuccessFactors uses basic auth, already set up
        # Test connection with a simple query
        test_url = f"{self.base_url}User"
        response = self.session.get(test_url, params={'$top': 1})
        response.raise_for_status()

        logger.info("SuccessFactors authentication successful")

    def disconnect(self) -> None:
        """Disconnect from SAP system"""
        if self.connection:
            self.connection.close()
            self.connection = None

        if self.session:
            self.session.close()
            self.session = None

        logger.info("Disconnected from SAP system")

    def get_sample_data(self, stream_name: str, stream_config: Dict) -> List[Dict]:
        """Get sample data for schema discovery"""
        logger.info(f"Getting sample data for stream: {stream_name}")

        try:
            if self.system_type in ["erp", "s4hana"] and self.use_rfc:
                return self._get_sample_data_rfc(stream_name, stream_config)
            else:
                return self._get_sample_data_http(stream_name, stream_config)
        except Exception as e:
            logger.error(f"Failed to get sample data for {stream_name}: {e}")
            return []

    def _get_sample_data_rfc(self, stream_name: str, stream_config: Dict) -> List[Dict]:
        """Get sample data using RFC"""
        table_name = stream_config.get("table_name", stream_name.upper())

        # Get table structure
        result = self.connection.call("RFC_READ_TABLE",
                                    QUERY_TABLE=table_name,
                                    ROWCOUNT=5)

        # Parse result
        data = []
        if result.get("DATA"):
            fields = [field["FIELDNAME"] for field in result.get("FIELDS", [])]

            for row in result["DATA"]:
                record = {}
                values = row["WA"].split("\t")
                for i, field in enumerate(fields):
                    if i < len(values):
                        record[field.lower()] = values[i].strip()
                data.append(record)

        return data

    def _get_sample_data_http(self, stream_name: str, stream_config: Dict) -> List[Dict]:
        """Get sample data using HTTP API"""
        endpoint = stream_config.get("endpoint", stream_name)
        url = f"{self.base_url}{endpoint}"

        params = {"$top": 5}
        if self.system_type == "successfactors":
            params["$format"] = "json"

        response = self.session.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract data based on system type
        if self.system_type == "successfactors":
            return data.get("d", {}).get("results", [])
        elif self.system_type == "business_one":
            return data.get("value", [])
        else:
            return data if isinstance(data, list) else [data]

    def get_records(self, stream_name: str, stream_config: Dict, start_date: Optional[str] = None) -> Iterator[Dict]:
        """Get records from SAP system"""
        logger.info(f"Getting records for stream: {stream_name}")

        if self.system_type in ["erp", "s4hana"] and self.use_rfc:
            yield from self._get_records_rfc(stream_name, stream_config, start_date)
        else:
            yield from self._get_records_http(stream_name, stream_config, start_date)

    def _get_records_rfc(self, stream_name: str, stream_config: Dict, start_date: Optional[str] = None) -> Iterator[Dict]:
        """Get records using RFC"""
        table_name = stream_config.get("table_name", stream_name.upper())
        date_field = stream_config.get("date_field")

        # Build WHERE clause for incremental sync
        where_clause = ""
        if start_date and date_field:
            # Convert date format for SAP
            sap_date = start_date.replace("-", "").split("T")[0]  # YYYYMMDD
            where_clause = f"{date_field} >= '{sap_date}'"

        # Get data in batches
        skip = 0
        while True:
            logger.debug(f"Fetching batch starting at {skip}")

            params = {
                "QUERY_TABLE": table_name,
                "ROWSKIPS": skip,
                "ROWCOUNT": self.batch_size
            }

            if where_clause:
                params["OPTIONS"] = [{"TEXT": where_clause}]

            result = self.connection.call("RFC_READ_TABLE", **params)

            if not result.get("DATA"):
                break

            # Parse and yield records
            fields = [field["FIELDNAME"] for field in result.get("FIELDS", [])]

            batch_count = 0
            for row in result["DATA"]:
                record = {}
                values = row["WA"].split("\t")
                for i, field in enumerate(fields):
                    if i < len(values):
                        value = values[i].strip()
                        # Convert SAP date format
                        if field.endswith("_DATE") or field.endswith("DAT"):
                            value = self._convert_sap_date(value)
                        record[field.lower()] = value

                yield record
                batch_count += 1

            if batch_count < self.batch_size:
                break

            skip += self.batch_size

    def _get_records_http(self, stream_name: str, stream_config: Dict, start_date: Optional[str] = None) -> Iterator[Dict]:
        """Get records using HTTP API"""
        endpoint = stream_config.get("endpoint", stream_name)
        base_url = f"{self.base_url}{endpoint}"
        date_field = stream_config.get("date_field")

        skip = 0
        while True:
            params = {
                "$skip": skip,
                "$top": self.batch_size
            }

            # Add date filter for incremental sync
            if start_date and date_field:
                filter_clause = f"{date_field} ge datetime'{start_date}'"
                params["$filter"] = filter_clause

            # System-specific parameters
            if self.system_type == "successfactors":
                params["$format"] = "json"

            response = self.session.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Extract records based on system type
            if self.system_type == "successfactors":
                records = data.get("d", {}).get("results", [])
            elif self.system_type == "business_one":
                records = data.get("value", [])
            else:
                records = data if isinstance(data, list) else [data]

            if not records:
                break

            for record in records:
                yield record

            if len(records) < self.batch_size:
                break

            skip += self.batch_size

    def _convert_sap_date(self, sap_date: str) -> Optional[str]:
        """Convert SAP date format to ISO format"""
        if not sap_date or sap_date == "00000000":
            return None

        try:
            # SAP date format: YYYYMMDD
            if len(sap_date) == 8:
                year = sap_date[:4]
                month = sap_date[4:6]
                day = sap_date[6:8]
                return f"{year}-{month}-{day}"
            else:
                return sap_date
        except (ValueError, IndexError):
            return sap_date

    def test_connection(self) -> bool:
        """Test connection to SAP system"""
        try:
            self.connect()

            # Test a simple operation
            if self.system_type in ["erp", "s4hana"] and self.use_rfc:
                # Test RFC connection
                result = self.connection.call("RFC_SYSTEM_INFO")
                logger.info(f"Connected to SAP system: {result.get('RFCSI_EXPORT', {}).get('RFCSYSID', 'Unknown')}")
            else:
                # Test HTTP connection (already done in connect)
                pass

            self.disconnect()
            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False