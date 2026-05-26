"""
Microsoft Dynamics Singer.io Tap Implementation
Main tap class for Microsoft Dynamics data extraction
"""

import singer
from singer import utils, metadata
from singer.catalog import Catalog, CatalogEntry, Schema
from typing import Dict, List, Any, Optional, Iterator
import sys
import json
from datetime import datetime, timezone
import requests
import logging

from .client import DynamicsClient
from .streams import STREAM_MAPS
from .schemas import DYNAMICS_SCHEMAS

logger = logging.getLogger(__name__)

REQUIRED_CONFIG_KEYS = [
    "dynamics_system_type",  # finance_operations, business_central, sales, customer_service, marketing, ax, nav
    "tenant_id",            # Azure AD tenant ID
    "client_id",            # Azure AD application ID
    "client_secret",        # Azure AD application secret
    "base_url"             # Dynamics instance URL
]

OPTIONAL_CONFIG_KEYS = [
    "api_version",          # API version (default: v9.2 for CRM, varies for others)
    "batch_size",          # Records per batch (default: 1000)
    "start_date",          # Initial sync start date
    "include_deleted",     # Include deleted records
    "environment_name",    # Environment name for Business Central
    "company_id",          # Company ID for Business Central
    "data_area_id",        # Data area ID for Finance & Operations
    "page_size",           # Page size for pagination
    "timeout",             # Request timeout in seconds
    "max_retries"          # Maximum number of retries
]


class TapDynamics:
    """Main Microsoft Dynamics Singer.io tap implementation"""

    def __init__(self, config: Dict[str, Any], catalog: Optional[Catalog] = None, state: Optional[Dict] = None):
        self.config = config
        self.catalog = catalog
        self.state = state or {}

        # Initialize Dynamics client
        self.client = DynamicsClient(config)

        # Validate required configuration
        self._validate_config()

        # Set up logging
        logging.basicConfig(level=logging.INFO)

    def _validate_config(self) -> None:
        """Validate required configuration parameters"""
        missing_keys = []
        for key in REQUIRED_CONFIG_KEYS:
            if key not in self.config:
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        # Validate Dynamics system type
        valid_types = [
            "finance_operations", "business_central", "sales", "customer_service",
            "marketing", "supply_chain", "ax", "nav"
        ]
        if self.config["dynamics_system_type"] not in valid_types:
            raise ValueError(f"Invalid dynamics_system_type. Must be one of: {valid_types}")

    def discover(self) -> Catalog:
        """Discover available streams and their schemas"""
        logger.info("Starting Dynamics stream discovery")

        catalog_entries = []
        system_type = self.config["dynamics_system_type"]

        # Get streams for the specific Dynamics system type
        streams = STREAM_MAPS.get(system_type, {})

        for stream_name, stream_config in streams.items():
            logger.info(f"Discovering stream: {stream_name}")

            try:
                # Get schema from predefined schemas or discover dynamically
                schema = self._get_stream_schema(stream_name, stream_config)

                # Create catalog entry
                catalog_entry = CatalogEntry(
                    tap_stream_id=stream_name,
                    stream=stream_name,
                    schema=schema,
                    key_properties=stream_config.get("key_properties", ["id"]),
                    replication_method=stream_config.get("replication_method", "INCREMENTAL"),
                    replication_key=stream_config.get("replication_key")
                )

                catalog_entries.append(catalog_entry)

            except Exception as e:
                logger.error(f"Failed to discover stream {stream_name}: {e}")
                continue

        logger.info(f"Discovered {len(catalog_entries)} streams")
        return Catalog(catalog_entries)

    def _get_stream_schema(self, stream_name: str, stream_config: Dict) -> Schema:
        """Get schema for a specific stream"""
        # First check if we have a predefined schema
        if stream_name in DYNAMICS_SCHEMAS:
            return Schema.from_dict(DYNAMICS_SCHEMAS[stream_name])

        # Otherwise, discover schema dynamically using metadata
        try:
            metadata_info = self.client.get_entity_metadata(stream_name, stream_config)
            schema_dict = self._convert_metadata_to_schema(metadata_info)
            return Schema.from_dict(schema_dict)
        except Exception as e:
            logger.warning(f"Could not discover schema for {stream_name}: {e}")
            # Return basic schema
            return Schema.from_dict({
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "createdon": {"type": "string", "format": "date-time"},
                    "modifiedon": {"type": "string", "format": "date-time"}
                }
            })

    def _convert_metadata_to_schema(self, metadata_info: Dict) -> Dict:
        """Convert Dynamics metadata to JSON Schema"""
        properties = {}

        for field in metadata_info.get("fields", []):
            field_name = field.get("name", "").lower()
            field_type = field.get("type", "string")

            # Map Dynamics types to JSON Schema types
            if field_type in ["Edm.String", "Edm.Guid"]:
                properties[field_name] = {"type": "string"}
            elif field_type == "Edm.Int32":
                properties[field_name] = {"type": "integer"}
            elif field_type == "Edm.Int64":
                properties[field_name] = {"type": "integer"}
            elif field_type in ["Edm.Decimal", "Edm.Double"]:
                properties[field_name] = {"type": "number"}
            elif field_type == "Edm.Boolean":
                properties[field_name] = {"type": "boolean"}
            elif field_type == "Edm.DateTimeOffset":
                properties[field_name] = {"type": "string", "format": "date-time"}
            elif field_type == "Edm.Date":
                properties[field_name] = {"type": "string", "format": "date"}
            else:
                properties[field_name] = {"type": "string"}

        return {
            "type": "object",
            "properties": properties
        }

    def sync(self) -> None:
        """Sync data from Dynamics system"""
        logger.info("Starting Dynamics data sync")

        if not self.catalog:
            logger.error("No catalog provided for sync")
            return

        # Connect to Dynamics system
        self.client.connect()

        try:
            selected_streams = []
            for catalog_entry in self.catalog.streams:
                stream_metadata = metadata.to_map(catalog_entry.metadata)
                if metadata.get(stream_metadata, (), "selected"):
                    selected_streams.append(catalog_entry)

            logger.info(f"Syncing {len(selected_streams)} selected streams")

            for catalog_entry in selected_streams:
                self._sync_stream(catalog_entry)

        finally:
            self.client.disconnect()

        logger.info("Dynamics data sync completed")

    def _sync_stream(self, catalog_entry: CatalogEntry) -> None:
        """Sync a specific stream"""
        stream_name = catalog_entry.tap_stream_id
        logger.info(f"Syncing stream: {stream_name}")

        # Write schema
        singer.write_schema(
            stream_name=stream_name,
            schema=catalog_entry.schema.to_dict(),
            key_properties=catalog_entry.key_properties
        )

        # Get stream configuration
        system_type = self.config["dynamics_system_type"]
        stream_config = STREAM_MAPS.get(system_type, {}).get(stream_name, {})

        # Get bookmark for incremental sync
        bookmark = self.state.get("bookmarks", {}).get(stream_name, {})
        start_date = bookmark.get("last_updated") or self.config.get("start_date")

        # Sync data
        try:
            records = self.client.get_records(stream_name, stream_config, start_date)

            record_count = 0
            max_bookmark = start_date

            for record in records:
                # Transform record if needed
                transformed_record = self._transform_record(record, catalog_entry.schema)

                # Write record
                singer.write_record(stream_name, transformed_record)
                record_count += 1

                # Update bookmark
                record_date = self._get_record_date(record, catalog_entry.replication_key)
                if record_date and (not max_bookmark or record_date > max_bookmark):
                    max_bookmark = record_date

                # Write state periodically
                if record_count % 1000 == 0:
                    self._write_state(stream_name, max_bookmark)

            # Write final state
            self._write_state(stream_name, max_bookmark)

            logger.info(f"Synced {record_count} records from {stream_name}")

        except Exception as e:
            logger.error(f"Failed to sync stream {stream_name}: {e}")
            raise

    def _transform_record(self, record: Dict, schema: Schema) -> Dict:
        """Transform record according to schema"""
        transformed = {}
        schema_props = schema.to_dict().get("properties", {})

        for field, value in record.items():
            field_lower = field.lower()

            if field_lower in schema_props:
                field_type = schema_props[field_lower].get("type")

                # Handle Dynamics-specific transformations
                if field.startswith("@odata."):
                    # Skip OData metadata fields
                    continue
                elif field.endswith("@OData.Community.Display.V1.FormattedValue"):
                    # Skip formatted value fields
                    continue
                elif field_type == "string" and value is not None:
                    transformed[field_lower] = str(value)
                elif field_type == "integer" and value is not None:
                    try:
                        transformed[field_lower] = int(value)
                    except (ValueError, TypeError):
                        transformed[field_lower] = None
                elif field_type == "number" and value is not None:
                    try:
                        transformed[field_lower] = float(value)
                    except (ValueError, TypeError):
                        transformed[field_lower] = None
                elif field_type == "boolean" and value is not None:
                    if isinstance(value, str):
                        transformed[field_lower] = value.lower() in ('true', '1', 'yes')
                    else:
                        transformed[field_lower] = bool(value)
                else:
                    transformed[field_lower] = value
            else:
                # Include unmapped fields as strings
                if not field.startswith("@odata."):
                    transformed[field.lower()] = str(value) if value is not None else None

        return transformed

    def _get_record_date(self, record: Dict, replication_key: Optional[str]) -> Optional[str]:
        """Get the date value for bookmark updates"""
        if not replication_key:
            return None

        # Try both original case and lowercase
        date_value = record.get(replication_key) or record.get(replication_key.lower())

        if isinstance(date_value, str):
            return date_value
        elif isinstance(date_value, datetime):
            return date_value.isoformat()
        else:
            return str(date_value) if date_value else None

    def _write_state(self, stream_name: str, bookmark_value: Optional[str]) -> None:
        """Write state with updated bookmark"""
        if bookmark_value:
            if "bookmarks" not in self.state:
                self.state["bookmarks"] = {}

            self.state["bookmarks"][stream_name] = {
                "last_updated": bookmark_value,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            singer.write_state(self.state)


def main():
    """Main entry point for the tap"""
    # Parse command line arguments
    args = utils.parse_args(REQUIRED_CONFIG_KEYS)

    # Initialize tap
    tap = TapDynamics(args.config, args.catalog, args.state)

    if args.discover:
        # Discovery mode
        catalog = tap.discover()
        catalog.dump()
    else:
        # Sync mode
        tap.sync()


if __name__ == "__main__":
    main()