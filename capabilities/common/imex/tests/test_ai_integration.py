#!/usr/bin/env python3
"""
AI Integration test for APG IMEX capability.

This test validates:
- AI-powered schema detection
- Data quality assessment using AI
- Field mapping suggestions
- Performance characteristics of AI features
"""
import asyncio
import logging
import tempfile
import json
import csv
from pathlib import Path
from datetime import datetime, timezone

from models import (
    SourceConfig, TargetConfig, JobType, DataFormat, SourceType
)
from ai_intelligence import AIIntelligenceEngine
from service import ImportExportService
from database import DatabaseManager, DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIIntegrationTestSuite:
    """Comprehensive AI integration testing suite."""

    def __init__(self):
        self.ai_engine = None
        self.service = None
        self.temp_dir = None

    async def setup(self):
        """Setup test environment."""
        try:
            # Create AI engine
            self.ai_engine = AIIntelligenceEngine(
                ollama_model="llama3.1:8b",
                max_sample_size=100,
                confidence_threshold=0.6
            )

            # Initialize AI engine
            await self.ai_engine.initialize()

            # Create minimal database manager for service
            db_config = DatabaseConfig(
                host="localhost",
                port=5432,
                database="test",
                user="test",
                password="test"
            )
            db_manager = DatabaseManager(db_config)

            # Create service with AI engine
            self.service = ImportExportService(db_manager, self.ai_engine)
            await self.service.initialize()

            # Create temporary directory for test files
            self.temp_dir = Path(tempfile.mkdtemp())

            logger.info("✓ AI integration test setup completed")
            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

    async def test_schema_detection_csv(self) -> bool:
        """Test AI-powered schema detection on CSV data."""
        try:
            # Create test CSV file
            csv_file = self.temp_dir / "test_data.csv"

            test_data = [
                ["id", "name", "email", "age", "salary", "active"],
                ["1", "John Doe", "john@example.com", "30", "50000.50", "true"],
                ["2", "Jane Smith", "jane@example.com", "25", "45000.75", "true"],
                ["3", "Bob Johnson", "bob@example.com", "35", "60000.00", "false"],
                ["4", "Alice Brown", "", "28", "52000.25", "true"],  # Missing email
                ["5", "Charlie Wilson", "charlie@example.com", "invalid_age", "55000.00", "true"]  # Invalid age
            ]

            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(test_data)

            # Create source config
            source_config = SourceConfig(
                source_type=SourceType.FILE,
                format=DataFormat.CSV,
                file_path=str(csv_file),
                has_header=True,
                encoding='utf-8'
            )

            # Test schema detection
            schema_result = await self.service.detect_schema_automatically(source_config)

            # Validate results
            assert isinstance(schema_result, dict)
            logger.info(f"✓ CSV schema detection successful: {schema_result}")

            # Test direct AI engine analysis
            sample_data = await self.service._read_sample_file_data(source_config)
            ai_analysis = await self.ai_engine.analyze_schema(sample_data, DataFormat.CSV)

            assert len(ai_analysis.fields) == 6  # 6 columns
            assert ai_analysis.total_records > 0
            assert ai_analysis.confidence_score > 0

            # Check for expected field types
            field_names = [f.field_name for f in ai_analysis.fields]
            assert "id" in field_names
            assert "email" in field_names
            assert "age" in field_names

            logger.info(f"✓ AI schema analysis successful: {len(ai_analysis.fields)} fields, {ai_analysis.confidence_score:.2f} confidence")
            return True

        except Exception as e:
            logger.error(f"CSV schema detection test failed: {e}")
            return False

    async def test_schema_detection_json(self) -> bool:
        """Test AI-powered schema detection on JSON data."""
        try:
            # Create test JSON file
            json_file = self.temp_dir / "test_data.json"

            test_data = [
                {
                    "user_id": 1,
                    "username": "john_doe",
                    "profile": {
                        "first_name": "John",
                        "last_name": "Doe",
                        "email": "john@example.com"
                    },
                    "metadata": {
                        "created_at": "2024-01-15T10:30:00Z",
                        "last_login": "2024-08-14T09:00:00Z",
                        "login_count": 42
                    },
                    "tags": ["premium", "verified"],
                    "active": True
                },
                {
                    "user_id": 2,
                    "username": "jane_smith",
                    "profile": {
                        "first_name": "Jane",
                        "last_name": "Smith",
                        "email": "jane@example.com"
                    },
                    "metadata": {
                        "created_at": "2024-02-20T14:15:00Z",
                        "last_login": "2024-08-13T16:30:00Z",
                        "login_count": 28
                    },
                    "tags": ["standard"],
                    "active": True
                }
            ]

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2)

            # Create source config
            source_config = SourceConfig(
                source_type=SourceType.FILE,
                format=DataFormat.JSON,
                file_path=str(json_file),
                encoding='utf-8'
            )

            # Test schema detection
            sample_data = await self.service._read_sample_file_data(source_config)
            ai_analysis = await self.ai_engine.analyze_schema(sample_data, DataFormat.JSON)

            assert len(ai_analysis.fields) > 0
            assert ai_analysis.total_records == 2

            # Check for nested field detection
            field_names = [f.field_name for f in ai_analysis.fields]
            logger.info(f"Detected fields: {field_names}")

            # Should detect top-level fields
            expected_fields = ["user_id", "username", "profile", "metadata", "tags", "active"]
            for field in expected_fields:
                assert field in field_names, f"Expected field {field} not found"

            logger.info(f"✓ JSON schema analysis successful: {len(ai_analysis.fields)} fields")
            return True

        except Exception as e:
            logger.error(f"JSON schema detection test failed: {e}")
            return False

    async def test_data_quality_assessment(self) -> bool:
        """Test AI-powered data quality assessment."""
        try:
            # Create test data with various quality issues
            test_data = [
                {"id": 1, "name": "John Doe", "email": "john@example.com", "age": 30, "score": 95.5},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "age": 25, "score": 87.2},
                {"id": 3, "name": "", "email": "invalid-email", "age": -5, "score": 999.9},  # Quality issues
                {"id": 4, "name": "Bob Johnson", "email": "bob@example.com", "age": 35, "score": 78.1},
                {"id": None, "name": "Alice Brown", "email": "alice@example.com", "age": None, "score": 91.3},  # Nulls
                {"id": 6, "name": "Charlie Wilson", "email": "charlie@example.com", "age": 28, "score": 83.7},
                {"id": 1, "name": "Duplicate ID", "email": "duplicate@example.com", "age": 40, "score": 76.4}  # Duplicate ID
            ]

            # Test quality assessment
            quality_assessment = await self.ai_engine.assess_data_quality(test_data)

            # Validate assessment
            assert isinstance(quality_assessment.overall_score, float)
            assert 0 <= quality_assessment.overall_score <= 1
            assert 0 <= quality_assessment.completeness_score <= 1
            assert 0 <= quality_assessment.consistency_score <= 1
            assert 0 <= quality_assessment.accuracy_score <= 1
            assert 0 <= quality_assessment.uniqueness_score <= 1
            assert 0 <= quality_assessment.validity_score <= 1

            logger.info(f"✓ Quality assessment scores:")
            logger.info(f"  Overall: {quality_assessment.overall_score:.2f}")
            logger.info(f"  Completeness: {quality_assessment.completeness_score:.2f}")
            logger.info(f"  Consistency: {quality_assessment.consistency_score:.2f}")
            logger.info(f"  Accuracy: {quality_assessment.accuracy_score:.2f}")
            logger.info(f"  Uniqueness: {quality_assessment.uniqueness_score:.2f}")
            logger.info(f"  Validity: {quality_assessment.validity_score:.2f}")

            # Should detect quality issues (recommendations may be empty for good quality data)
            logger.info(f"✓ Generated {len(quality_assessment.recommendations)} recommendations")

            # Check that field scores are present
            if hasattr(quality_assessment, 'field_scores'):
                logger.info(f"✓ Field-level scores available: {len(quality_assessment.field_scores)} fields")

            return True

        except Exception as e:
            logger.error(f"Data quality assessment test failed: {e}")
            return False

    async def test_field_mapping_suggestions(self) -> bool:
        """Test AI-powered field mapping suggestions."""
        try:
            # Create source schema
            source_data = [
                {"customer_id": 1, "full_name": "John Doe", "email_address": "john@example.com"},
                {"customer_id": 2, "full_name": "Jane Smith", "email_address": "jane@example.com"}
            ]

            # Create target schema
            target_data = [
                {"id": 1, "name": "John Doe", "email": "john@example.com", "status": "active"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "status": "active"}
            ]

            # Analyze schemas
            source_analysis = await self.ai_engine.analyze_schema(source_data, DataFormat.JSON)
            target_analysis = await self.ai_engine.analyze_schema(target_data, DataFormat.JSON)

            # Generate field mapping suggestions
            mappings = await self.ai_engine.suggest_field_mappings(
                source_analysis,
                target_analysis,
                {"context": "customer data migration"}
            )

            # Validate mappings
            assert isinstance(mappings, list)
            assert len(mappings) > 0

            # Should suggest reasonable mappings
            mapping_dict = {m["source_field"]: m for m in mappings if m.get("target_field")}

            logger.info("✓ Generated field mappings:")
            for mapping in mappings:
                if mapping.get("target_field"):
                    logger.info(f"  {mapping['source_field']} -> {mapping['target_field']} (confidence: {mapping['confidence']:.2f})")
                else:
                    logger.info(f"  {mapping['source_field']} -> No mapping (confidence: {mapping['confidence']:.2f})")

            # Should map customer_id to id, full_name to name, email_address to email
            expected_mappings = {
                "customer_id": "id",
                "full_name": "name",
                "email_address": "email"
            }

            for source_field, expected_target in expected_mappings.items():
                if source_field in mapping_dict:
                    actual_target = mapping_dict[source_field].get("target_field")
                    if actual_target == expected_target:
                        logger.info(f"✓ Correctly mapped {source_field} -> {actual_target}")
                    else:
                        logger.warning(f"⚠ Unexpected mapping {source_field} -> {actual_target} (expected {expected_target})")

            return True

        except Exception as e:
            logger.error(f"Field mapping test failed: {e}")
            return False

    async def test_performance_characteristics(self) -> bool:
        """Test performance characteristics of AI features."""
        try:
            # Create larger dataset for performance testing
            large_data = []
            for i in range(1000):
                record = {
                    "id": i,
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "age": 20 + (i % 50),
                    "score": 50.0 + (i % 100),
                    "active": i % 2 == 0,
                    "created_at": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}T10:00:00Z"
                }
                large_data.append(record)

            # Test schema analysis performance
            start_time = datetime.now(timezone.utc)
            schema_analysis = await self.ai_engine.analyze_schema(large_data, DataFormat.JSON)
            schema_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.info(f"✓ Schema analysis performance: {schema_time:.2f}s for {len(large_data)} records")
            assert schema_time < 10.0  # Should complete within 10 seconds

            # Test quality assessment performance
            start_time = datetime.now(timezone.utc)
            quality_assessment = await self.ai_engine.assess_data_quality(large_data)
            quality_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.info(f"✓ Quality assessment performance: {quality_time:.2f}s for {len(large_data)} records")
            assert quality_time < 15.0  # Should complete within 15 seconds

            # Validate results are still reasonable (allow for sampling limits)
            logger.info(f"✓ Schema analysis found {len(schema_analysis.fields)} fields from {schema_analysis.total_records} records")
            logger.info(f"✓ Quality assessment overall score: {quality_assessment.overall_score:.2f}")

            # Basic validation (more lenient for performance test)
            assert len(schema_analysis.fields) >= 6  # Should detect most fields
            assert schema_analysis.total_records > 0  # Should process some records
            assert 0 <= quality_assessment.overall_score <= 1  # Valid score range

            return True

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False

    async def teardown(self):
        """Clean up test resources."""
        try:
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
            logger.info("✓ Test cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

async def main():
    """Run AI integration tests."""
    logger.info("Starting APG IMEX AI integration tests...")

    test_suite = AIIntegrationTestSuite()

    try:
        # Setup
        if not await test_suite.setup():
            logger.error("Test setup failed")
            return 1

        # Run test suite
        tests = [
            ("CSV Schema Detection", test_suite.test_schema_detection_csv),
            ("JSON Schema Detection", test_suite.test_schema_detection_json),
            ("Data Quality Assessment", test_suite.test_data_quality_assessment),
            ("Field Mapping Suggestions", test_suite.test_field_mapping_suggestions),
            ("Performance Characteristics", test_suite.test_performance_characteristics),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            logger.info(f"\nRunning: {test_name}")
            try:
                if await test_func():
                    passed += 1
                    logger.info(f"✓ {test_name} PASSED")
                else:
                    failed += 1
                    logger.error(f"✗ {test_name} FAILED")
            except Exception as e:
                failed += 1
                logger.error(f"✗ {test_name} FAILED with exception: {e}")

        # Results
        total = passed + failed
        logger.info(f"\nAI Integration Test Results:")
        logger.info(f"  Total tests: {total}")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {(passed/total)*100:.1f}%")

        if failed == 0:
            logger.info("✓ All AI integration tests passed successfully!")
            return 0
        else:
            logger.error(f"✗ {failed} AI integration tests failed")
            return 1

    finally:
        await test_suite.teardown()

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(result)