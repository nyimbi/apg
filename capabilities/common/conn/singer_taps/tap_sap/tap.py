"""
SAP Singer.io Tap Implementation
Main tap class for SAP data extraction with comprehensive ERP support
"""

import singer
from singer import utils, metadata
from singer.catalog import Catalog, CatalogEntry, Schema
from typing import Dict, List, Any, Optional, Iterator
import sys
import json
from datetime import datetime, timezone
import pyrfc
import requests
import logging

from .client import SAPClient
from .streams import STREAM_MAPS
from .schemas import SAP_SCHEMAS

logger = logging.getLogger(__name__)

REQUIRED_CONFIG_KEYS = [
    "sap_system_type",  # erp, s4hana, business_one, successfactors, concur, ariba
    "host",
    "username",
    "password"
]

OPTIONAL_CONFIG_KEYS = [
    "client",           # SAP client number (for ERP/S/4HANA)
    "system_number",    # SAP system number
    "router",           # SAP router string
    "language",         # Language (default: EN)
    "api_version",      # API version
    "batch_size",       # Records per batch (default: 1000)
    "start_date",       # Initial sync start date
    "include_deleted",  # Include deleted records
    "custom_fields",    # Additional custom fields to extract
    "company_codes",    # Specific company codes to extract
    "plants",           # Specific plants to extract
]


class TapSAP:
    """Main SAP Singer.io tap implementation"""

    def __init__(self, config: Dict[str, Any], catalog: Optional[Catalog] = None, state: Optional[Dict] = None):
        self.config = config
        self.catalog = catalog
        self.state = state or {}

        # Initialize SAP client
        self.client = SAPClient(config)

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

        # Validate SAP system type
        valid_types = ["erp", "s4hana", "business_one", "successfactors", "concur", "ariba", "fieldglass"]
        if self.config["sap_system_type"] not in valid_types:
            raise ValueError(f"Invalid sap_system_type. Must be one of: {valid_types}")

    def discover(self) -> Catalog:
        """Discover available streams and their schemas"""
        logger.info("Starting SAP stream discovery")

        catalog_entries = []
        system_type = self.config["sap_system_type"]

        # Get streams for the specific SAP system type
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
        if stream_name in SAP_SCHEMAS:
            return Schema.from_dict(SAP_SCHEMAS[stream_name])

        # Otherwise, discover schema dynamically
        try:
            sample_data = self.client.get_sample_data(stream_name, stream_config)
            schema_dict = self._infer_schema_from_sample(sample_data)
            return Schema.from_dict(schema_dict)
        except Exception as e:
            logger.warning(f"Could not discover schema for {stream_name}: {e}")
            # Return basic schema
            return Schema.from_dict({
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "created_date": {"type": "string", "format": "date-time"},
                    "modified_date": {"type": "string", "format": "date-time"}
                }
            })

    def _infer_schema_from_sample(self, sample_data: List[Dict]) -> Dict:
        """Infer schema from sample data"""
        if not sample_data:
            return {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}
                }
            }

        # Analyze first few records to infer types
        properties = {}
        sample_records = sample_data[:10]  # Use first 10 records for inference

        for record in sample_records:
            for field, value in record.items():
                if field not in properties:
                    properties[field] = {"type": self._infer_type(value)}

        return {
            "type": "object",
            "properties": properties
        }

    def _infer_type(self, value: Any) -> str:
        """Infer JSON Schema type from value"""
        if value is None:
            return "string"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            # Check if it looks like a date
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return "string"
            except:
                return "string"
        elif isinstance(value, (list, tuple)):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"

    def sync(self) -> None:
        """Sync data from SAP system"""
        logger.info("Starting SAP data sync")

        if not self.catalog:
            logger.error("No catalog provided for sync")
            return

        # Connect to SAP system
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

        logger.info("SAP data sync completed")

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
        system_type = self.config["sap_system_type"]
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
            if field in schema_props:
                field_type = schema_props[field].get("type")

                # Type conversion based on schema
                if field_type == "string" and value is not None:
                    transformed[field] = str(value)
                elif field_type == "integer" and value is not None:
                    try:
                        transformed[field] = int(float(value))
                    except (ValueError, TypeError):
                        transformed[field] = None
                elif field_type == "number" and value is not None:
                    try:
                        transformed[field] = float(value)
                    except (ValueError, TypeError):
                        transformed[field] = None
                elif field_type == "boolean" and value is not None:
                    if isinstance(value, str):
                        transformed[field] = value.lower() in ('true', '1', 'yes', 'x')
                    else:
                        transformed[field] = bool(value)
                else:
                    transformed[field] = value
            else:
                transformed[field] = value

        return transformed

    def _get_record_date(self, record: Dict, replication_key: Optional[str]) -> Optional[str]:
        """Get the date value for bookmark updates"""
        if not replication_key or replication_key not in record:
            return None

        date_value = record[replication_key]
        if isinstance(date_value, str):
            return date_value
        elif isinstance(date_value, datetime):
            return date_value.isoformat()
        else:
            return str(date_value)

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
    tap = TapSAP(args.config, args.catalog, args.state)

    if args.discover:
        # Discovery mode
        catalog = tap.discover()
        catalog.dump()
    else:
        # Sync mode
        tap.sync()


if __name__ == "__main__":
    main()