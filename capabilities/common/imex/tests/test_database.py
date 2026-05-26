#!/usr/bin/env python3
"""
Database layer validation test for APG IMEX capability.

This test validates:
- Database schema creation and migration
- CRUD operations for all models
- Transaction management
- Connection pooling and health monitoring
- Performance characteristics
"""
import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone

from models import (
    ImportExportJob, JobExecution, SourceConfig, TargetConfig,
    JobStatus, JobType, DataFormat, SourceType, ValidationLevel,
    ErrorHandlingStrategy, ProcessingPriority, ProcessingMetrics
)
from database import DatabaseManager, DatabaseConfig, create_database_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseTestSuite:
    """Comprehensive database testing suite."""

    def __init__(self):
        self.db_manager = None
        self.test_job_id = None
        self.test_execution_id = None

    async def setup_test_database(self) -> bool:
        """
        Setup test database connection.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Check if PostgreSQL environment variables are set
            db_host = os.getenv('POSTGRES_HOST', 'localhost')
            db_port = int(os.getenv('POSTGRES_PORT', '5432'))
            db_name = os.getenv('POSTGRES_DB', 'imex_test')
            db_user = os.getenv('POSTGRES_USER', 'postgres')
            db_password = os.getenv('POSTGRES_PASSWORD', 'postgres')

            logger.info(f"Attempting to connect to PostgreSQL at {db_host}:{db_port}")

            # Create database manager
            self.db_manager = await create_database_manager(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password,
                min_size=2,
                max_size=5
            )

            logger.info("✓ Database connection established successfully")
            return True

        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            logger.info("Skipping database tests (database not available)")
            return False

    async def test_database_health(self) -> bool:
        """Test database health monitoring."""
        try:
            # Test health check
            health = await self.db_manager.health_check()
            logger.info(f"✓ Database health check: {health.is_healthy}")

            assert health.is_healthy == True
            assert health.total_connections > 0
            assert health.response_time_ms > 0

            # Test health check caching
            health2 = await self.db_manager.health_check()
            assert health2.last_check >= health.last_check

            logger.info("✓ Database health monitoring validated")
            return True

        except Exception as e:
            logger.error(f"✗ Database health test failed: {e}")
            return False

    async def test_job_crud_operations(self) -> bool:
        """Test complete CRUD operations for jobs."""
        try:
            # Create test job
            job_data = {
                "tenant_id": "test_tenant_db",
                "name": "Database Test Job",
                "description": "Test job for database validation",
                "job_type": JobType.IMPORT,
                "priority": ProcessingPriority.NORMAL,
                "source_config": SourceConfig(
                    source_type=SourceType.FILE,
                    format=DataFormat.CSV,
                    file_path="/tmp/test_db.csv",
                    chunk_size=1000
                ),
                "target_config": TargetConfig(
                    target_type=SourceType.DATABASE,
                    format=DataFormat.CSV,
                    database_config={"host": "localhost", "port": 5432},
                    batch_size=500
                ),
                "validation_level": ValidationLevel.BASIC,
                "error_handling": ErrorHandlingStrategy.LOG_AND_CONTINUE,
                "created_by": "db_test_user"
            }

            # Test job creation
            job = await self.db_manager.create_job(job_data)
            self.test_job_id = job.id
            logger.info(f"✓ Job created: {job.id}")

            # Test job retrieval
            retrieved_job = await self.db_manager.get_job(job.id)
            assert retrieved_job is not None
            assert retrieved_job.id == job.id
            assert retrieved_job.name == job.name
            assert retrieved_job.tenant_id == job.tenant_id
            logger.info("✓ Job retrieval successful")

            # Test job update
            updates = {
                "status": JobStatus.RUNNING,
                "updated_by": "db_test_system",
                "last_run_at": datetime.now(timezone.utc)
            }
            update_success = await self.db_manager.update_job(job.id, updates)
            assert update_success == True
            logger.info("✓ Job update successful")

            # Verify update
            updated_job = await self.db_manager.get_job(job.id)
            assert updated_job.status == JobStatus.RUNNING
            assert updated_job.last_run_at is not None
            logger.info("✓ Job update verification successful")

            # Test job listing
            jobs = await self.db_manager.list_jobs(
                tenant_id="test_tenant_db",
                limit=10
            )
            assert len(jobs) >= 1
            assert any(j.id == job.id for j in jobs)
            logger.info(f"✓ Job listing successful: {len(jobs)} jobs found")

            logger.info("✓ All job CRUD operations validated")
            return True

        except Exception as e:
            logger.error(f"✗ Job CRUD operations failed: {e}")
            return False

    async def test_execution_operations(self) -> bool:
        """Test job execution tracking operations."""
        try:
            if not self.test_job_id:
                logger.error("No test job available for execution test")
                return False

            # Create test execution
            execution_data = {
                "job_id": self.test_job_id,
                "execution_number": 1,
                "status": JobStatus.RUNNING,
                "started_at": datetime.now(timezone.utc),
                "metrics": ProcessingMetrics(
                    records_processed=1000,
                    records_successful=950,
                    records_failed=50,
                    processing_time_seconds=45.5,
                    throughput_records_per_second=22.0
                ),
                "worker_node": "test-worker-1",
                "execution_config": {"test_mode": True}
            }

            # Test execution creation
            execution = await self.db_manager.create_execution(execution_data)
            self.test_execution_id = execution.id
            logger.info(f"✓ Execution created: {execution.id}")

            # Test execution update
            execution_updates = {
                "status": JobStatus.COMPLETED,
                "completed_at": datetime.now(timezone.utc),
                "metrics": ProcessingMetrics(
                    records_processed=1000,
                    records_successful=980,
                    records_failed=20,
                    processing_time_seconds=50.0,
                    throughput_records_per_second=20.0
                )
            }
            update_success = await self.db_manager.update_execution(
                execution.id, execution_updates
            )
            assert update_success == True
            logger.info("✓ Execution update successful")

            # Test execution history retrieval
            executions = await self.db_manager.get_job_executions(self.test_job_id)
            assert len(executions) >= 1
            assert any(e.id == execution.id for e in executions)
            logger.info(f"✓ Execution history retrieval: {len(executions)} executions")

            logger.info("✓ All execution operations validated")
            return True

        except Exception as e:
            logger.error(f"✗ Execution operations failed: {e}")
            return False

    async def test_transaction_management(self) -> bool:
        """Test database transaction management."""
        try:
            # Test successful transaction
            async with self.db_manager.transaction() as tx:
                job_data = {
                    "tenant_id": "tx_test_tenant",
                    "name": "Transaction Test Job",
                    "job_type": JobType.EXPORT,
                    "source_config": SourceConfig(
                        source_type=SourceType.DATABASE,
                        format=DataFormat.JSON,
                        database_config={"host": "localhost"}
                    ),
                    "target_config": TargetConfig(
                        target_type=SourceType.FILE,
                        format=DataFormat.JSON,
                        file_path="/tmp/tx_test.json"
                    ),
                    "created_by": "tx_test_user"
                }

                job = await self.db_manager.create_job(job_data, tx)
                execution_data = {
                    "job_id": job.id,
                    "execution_number": 1,
                    "metrics": ProcessingMetrics()
                }
                execution = await self.db_manager.create_execution(execution_data, tx)

                # Transaction should commit automatically
                logger.info("✓ Transaction commit successful")

            # Verify data was committed
            retrieved_job = await self.db_manager.get_job(job.id)
            assert retrieved_job is not None
            logger.info("✓ Transaction data persistence verified")

            # Test transaction rollback
            rollback_job_id = None
            try:
                async with self.db_manager.transaction() as tx:
                    job_data["name"] = "Rollback Test Job"
                    rollback_job = await self.db_manager.create_job(job_data, tx)
                    rollback_job_id = rollback_job.id

                    # Force an error to trigger rollback
                    raise ValueError("Intentional error for rollback test")

            except ValueError:
                logger.info("✓ Transaction rollback triggered correctly")

            # Verify rollback - job should not exist
            if rollback_job_id:
                rolled_back_job = await self.db_manager.get_job(rollback_job_id)
                assert rolled_back_job is None
                logger.info("✓ Transaction rollback verified")

            logger.info("✓ All transaction management tests passed")
            return True

        except Exception as e:
            logger.error(f"✗ Transaction management failed: {e}")
            return False

    async def test_performance_characteristics(self) -> bool:
        """Test database performance characteristics."""
        try:
            # Test bulk operations
            start_time = datetime.now(timezone.utc)

            job_ids = []
            for i in range(10):
                job_data = {
                    "tenant_id": "perf_test_tenant",
                    "name": f"Performance Test Job {i+1}",
                    "job_type": JobType.IMPORT,
                    "source_config": SourceConfig(
                        source_type=SourceType.FILE,
                        format=DataFormat.CSV,
                        file_path=f"/tmp/perf_test_{i}.csv"
                    ),
                    "target_config": TargetConfig(
                        target_type=SourceType.DATABASE,
                        format=DataFormat.CSV,
                        database_config={"host": "localhost"}
                    ),
                    "created_by": "perf_test_user"
                }

                job = await self.db_manager.create_job(job_data)
                job_ids.append(job.id)

            bulk_create_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"✓ Bulk create performance: {bulk_create_time:.2f}s for 10 jobs")

            # Test bulk retrieval
            start_time = datetime.now(timezone.utc)
            jobs = await self.db_manager.list_jobs(
                tenant_id="perf_test_tenant",
                limit=50
            )
            bulk_read_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"✓ Bulk read performance: {bulk_read_time:.2f}s for {len(jobs)} jobs")

            # Validate performance thresholds
            assert bulk_create_time < 5.0  # Should create 10 jobs in under 5 seconds
            assert bulk_read_time < 1.0    # Should read jobs in under 1 second

            logger.info("✓ Performance characteristics validated")
            return True

        except Exception as e:
            logger.error(f"✗ Performance test failed: {e}")
            return False

    async def cleanup_test_data(self) -> bool:
        """Clean up test data."""
        try:
            # Delete test executions and jobs
            test_tenants = ["test_tenant_db", "tx_test_tenant", "perf_test_tenant"]

            for tenant in test_tenants:
                jobs = await self.db_manager.list_jobs(tenant_id=tenant, limit=100)
                for job in jobs:
                    success = await self.db_manager.delete_job(job.id)
                    if success:
                        logger.debug(f"Deleted test job: {job.id}")

            logger.info("✓ Test data cleanup completed")
            return True

        except Exception as e:
            logger.warning(f"Test cleanup failed: {e}")
            return True  # Don't fail overall test for cleanup issues

    async def teardown(self):
        """Teardown database connection."""
        if self.db_manager:
            await self.db_manager.close()
            logger.info("✓ Database connection closed")

async def main():
    """Run comprehensive database tests."""
    logger.info("Starting APG IMEX database validation tests...")

    test_suite = DatabaseTestSuite()

    try:
        # Setup database connection
        if not await test_suite.setup_test_database():
            logger.info("Database tests skipped - no database connection available")
            return 0

        # Run test suite
        tests = [
            ("Database Health Check", test_suite.test_database_health),
            ("Job CRUD Operations", test_suite.test_job_crud_operations),
            ("Execution Operations", test_suite.test_execution_operations),
            ("Transaction Management", test_suite.test_transaction_management),
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

        # Cleanup
        await test_suite.cleanup_test_data()

        # Results
        total = passed + failed
        logger.info(f"\nDatabase Test Results:")
        logger.info(f"  Total tests: {total}")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {(passed/total)*100:.1f}%")

        if failed == 0:
            logger.info("✓ All database tests passed successfully!")
            return 0
        else:
            logger.error(f"✗ {failed} database tests failed")
            return 1

    finally:
        await test_suite.teardown()

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(result)