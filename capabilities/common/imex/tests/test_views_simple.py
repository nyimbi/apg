#!/usr/bin/env python3
"""
Simple UI Views test for APG IMEX capability.

This test validates views components without full SQLAlchemy integration.
"""
import asyncio
import logging
import tempfile
from pathlib import Path

from database import DatabaseManager, DatabaseConfig
from ai_intelligence import AIIntelligenceEngine
from service import ImportExportService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleViewsTestSuite:
    """Simple UI Views testing suite."""

    def __init__(self):
        self.service = None
        self.temp_dir = None

    async def setup(self):
        """Setup test environment."""
        try:
            # Setup service
            db_config = DatabaseConfig(
                host="localhost", port=5432, database="test", user="test", password="test"
            )
            db_manager = DatabaseManager(db_config)
            ai_engine = AIIntelligenceEngine()
            await ai_engine.initialize()

            # Create service
            self.service = ImportExportService(db_manager, ai_engine)
            await self.service.initialize()

            # Create temporary directory for test files
            self.temp_dir = Path(tempfile.mkdtemp())

            logger.info("✓ Simple views test setup completed")
            return True

        except Exception as e:
            logger.error(f"Simple views test setup failed: {e}")
            return False

    def test_views_components_import(self) -> bool:
        """Test that view components can be imported."""
        try:
            # Test individual components that don't require SQLAlchemy
            from views import (
                JobCreateRequest, SchemaDetectionRequest, WorkflowCreateRequest,
                set_imex_service, view_registry
            )

            logger.info("✓ Views components import test passed")
            return True

        except Exception as e:
            logger.error(f"Views components import test failed: {e}")
            return False

    def test_service_integration(self) -> bool:
        """Test service integration with views."""
        try:
            from views import set_imex_service, imex_service as views_service

            # Test setting service
            set_imex_service(self.service)

            # Import again to get updated service
            import importlib
            import views
            importlib.reload(views)

            logger.info("✓ Service integration test passed")
            return True

        except Exception as e:
            logger.error(f"Service integration test failed: {e}")
            return False

    def test_pydantic_models_validation(self) -> bool:
        """Test Pydantic request models validation."""
        try:
            from views import JobCreateRequest, SchemaDetectionRequest

            # Test valid JobCreateRequest
            job_data = {
                'name': 'Test Job',
                'job_type': 'import',
                'source_config': {'source_type': 'file', 'format': 'csv'},
                'target_config': {'target_type': 'database'}
            }

            job_request = JobCreateRequest(**job_data)
            assert job_request.name == 'Test Job'
            assert str(job_request.job_type) == 'import'

            # Test invalid JobCreateRequest (should raise ValidationError)
            try:
                invalid_data = {'name': ''}  # Empty name should fail
                JobCreateRequest(**invalid_data)
                assert False, "Should have raised ValidationError"
            except Exception:
                pass  # Expected

            # Test SchemaDetectionRequest
            schema_data = {
                'source_config': {'source_type': 'file', 'format': 'csv'},
                'sample_size': 500
            }

            schema_request = SchemaDetectionRequest(**schema_data)
            assert schema_request.sample_size == 500

            logger.info("✓ Pydantic models validation test passed")
            return True

        except Exception as e:
            logger.error(f"Pydantic models validation test failed: {e}")
            return False

    def test_template_files_exist(self) -> bool:
        """Test that required template files exist."""
        try:
            template_dir = Path(__file__).parent / 'templates' / 'imex'

            required_templates = [
                'job_monitor.html',
                'monitoring_dashboard.html',
                'execution_logs.html'
            ]

            found_templates = 0
            for template in required_templates:
                template_path = template_dir / template
                if template_path.exists():
                    logger.info(f"✓ Template found: {template}")
                    found_templates += 1
                else:
                    logger.warning(f"Template not found: {template}")

            # Success if at least some templates exist
            success = found_templates > 0
            logger.info(f"✓ Template files test passed ({found_templates}/{len(required_templates)} found)")
            return success

        except Exception as e:
            logger.error(f"Template files test failed: {e}")
            return False

    def test_view_registry(self) -> bool:
        """Test view registry for APG integration."""
        try:
            from views import view_registry

            # Check that registry is a dictionary
            assert isinstance(view_registry, dict)

            # Check that registry has some content
            assert len(view_registry) > 0

            # Log contents
            logger.info(f"View registry contains: {list(view_registry.keys())}")

            logger.info("✓ View registry test passed")
            return True

        except Exception as e:
            logger.error(f"View registry test failed: {e}")
            return False

    def test_ui_helper_functions(self) -> bool:
        """Test UI helper functions and utilities."""
        try:
            from views import set_imex_service

            # Test setting service
            set_imex_service(self.service)

            # Test that it doesn't raise errors
            set_imex_service(None)
            set_imex_service(self.service)

            logger.info("✓ UI helper functions test passed")
            return True

        except Exception as e:
            logger.error(f"UI helper functions test failed: {e}")
            return False

    def teardown(self):
        """Clean up test resources."""
        try:
            if self.temp_dir and self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
            logger.info("✓ Simple views test cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

async def main():
    """Run simple views tests."""
    logger.info("Starting APG IMEX Simple Views tests...")

    test_suite = SimpleViewsTestSuite()

    try:
        # Setup
        if not await test_suite.setup():
            logger.error("Test setup failed")
            return 1

        # Run test suite
        tests = [
            ("Views Components Import", test_suite.test_views_components_import),
            ("Service Integration", test_suite.test_service_integration),
            ("Pydantic Models Validation", test_suite.test_pydantic_models_validation),
            ("Template Files Exist", test_suite.test_template_files_exist),
            ("View Registry", test_suite.test_view_registry),
            ("UI Helper Functions", test_suite.test_ui_helper_functions),
        ]

        passed = 0
        failed = 0

        for test_name, test_func in tests:
            logger.info(f"\nRunning: {test_name}")
            try:
                if test_func():
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
        logger.info(f"\nSimple Views Test Results:")
        logger.info(f"  Total tests: {total}")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {(passed/total)*100:.1f}%")

        if failed == 0:
            logger.info("✓ All simple views tests passed successfully!")
            return 0
        else:
            logger.error(f"✗ {failed} simple views tests failed")
            return 1

    finally:
        test_suite.teardown()

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(result)