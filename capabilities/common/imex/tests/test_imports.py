#!/usr/bin/env python3
"""
Test script to validate all imports work correctly.
"""
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)

def test_imports():
    """Test all module imports."""
    try:
        # Test models import
        from models import (
            ImportExportJob, JobExecution, SourceConfig, TargetConfig,
            JobStatus, JobType, DataFormat, ValidationLevel
        )
        print("✓ Models import successful")

        # Test database import
        from database import DatabaseManager, DatabaseConfig, HealthStatus
        print("✓ Database import successful")

        # Test service import
        from service import ImportExportService
        print("✓ Service import successful")

        # Test basic instantiation
        job_data = {
            "tenant_id": "test_tenant",
            "name": "Test Job",
            "job_type": JobType.IMPORT,
            "source_config": SourceConfig(
                source_type="file",
                format=DataFormat.CSV,
                file_path="/tmp/test.csv"
            ),
            "target_config": TargetConfig(
                target_type="database",
                format=DataFormat.CSV,
                database_config={"host": "localhost"}
            ),
            "created_by": "test_user"
        }

        job = ImportExportJob(**job_data)
        print(f"✓ Job creation successful: {job.id}")

        print("✓ All imports and basic functionality verified")
        return True

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)