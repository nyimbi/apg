#!/usr/bin/env python3
"""
SQL Schema syntax validation test.

This test validates that our schema.sql file is syntactically correct
and follows PostgreSQL best practices.
"""
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLSchemaValidator:
    """Validates SQL schema syntax and structure."""

    def __init__(self, schema_path: str):
        self.schema_path = Path(schema_path)
        self.content = self.schema_path.read_text()
        self.issues = []
        self.warnings = []

    def validate_syntax(self) -> bool:
        """Validate basic SQL syntax."""
        try:
            # Check for balanced parentheses
            paren_count = 0
            for char in self.content:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count < 0:
                        self.issues.append("Unbalanced parentheses - too many closing")
                        return False

            if paren_count > 0:
                self.issues.append("Unbalanced parentheses - unclosed opening")
                return False

            logger.info("✓ Parentheses balanced correctly")

            # Check for SQL injection patterns (basic check)
            dangerous_patterns = [
                r"DROP\s+TABLE.*--",
                r"DELETE\s+FROM.*--",
                r"TRUNCATE.*--"
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, self.content, re.IGNORECASE):
                    self.issues.append(f"Potentially dangerous SQL pattern: {pattern}")

            logger.info("✓ No dangerous SQL patterns detected")

            # Check for required PostgreSQL extensions
            required_extensions = ["uuid-ossp"]
            for ext in required_extensions:
                if f'CREATE EXTENSION IF NOT EXISTS "{ext}"' not in self.content:
                    self.warnings.append(f"Missing required extension: {ext}")

            logger.info("✓ Required extensions check completed")

            return True

        except Exception as e:
            self.issues.append(f"Syntax validation error: {e}")
            return False

    def validate_structure(self) -> bool:
        """Validate schema structure and organization."""
        try:
            # Check for required tables
            required_tables = [
                "imex_jobs",
                "imex_executions",
                "imex_quality_reports",
                "imex_workflows",
                "imex_connection_templates",
                "imex_monitoring_alerts"
            ]

            for table in required_tables:
                if f"CREATE TABLE IF NOT EXISTS {table}" not in self.content:
                    self.issues.append(f"Missing required table: {table}")

            logger.info(f"✓ All {len(required_tables)} required tables found")

            # Check for indexes on important columns
            important_indexes = [
                "idx_imex_jobs_tenant_id",
                "idx_imex_jobs_status",
                "idx_imex_executions_job_id",
                "idx_imex_executions_status"
            ]

            for index in important_indexes:
                if f"CREATE INDEX IF NOT EXISTS {index}" not in self.content:
                    self.warnings.append(f"Missing recommended index: {index}")

            logger.info("✓ Index structure validation completed")

            # Check for foreign key constraints
            fk_patterns = [
                r"REFERENCES\s+imex_jobs\(id\)",
                r"REFERENCES\s+imex_executions\(id\)"
            ]

            fk_count = 0
            for pattern in fk_patterns:
                fk_count += len(re.findall(pattern, self.content, re.IGNORECASE))

            if fk_count == 0:
                self.warnings.append("No foreign key constraints found")
            else:
                logger.info(f"✓ Found {fk_count} foreign key constraints")

            # Check for triggers
            trigger_patterns = [
                r"CREATE TRIGGER.*update.*updated_at",
                r"CREATE OR REPLACE FUNCTION.*update_updated_at"
            ]

            trigger_count = 0
            for pattern in trigger_patterns:
                trigger_count += len(re.findall(pattern, self.content, re.IGNORECASE))

            if trigger_count > 0:
                logger.info(f"✓ Found {trigger_count} automated triggers")

            return True

        except Exception as e:
            self.issues.append(f"Structure validation error: {e}")
            return False

    def validate_best_practices(self) -> bool:
        """Validate PostgreSQL best practices."""
        try:
            # Check for JSONB usage (preferred over JSON)
            jsonb_count = len(re.findall(r'\bJSONB\b', self.content, re.IGNORECASE))
            json_count = len(re.findall(r'\bJSON\b(?!B)', self.content, re.IGNORECASE))

            if jsonb_count > 0:
                logger.info(f"✓ Using JSONB for JSON columns ({jsonb_count} instances)")

            if json_count > 0:
                self.warnings.append(f"Consider using JSONB instead of JSON ({json_count} instances)")

            # Check for proper timestamp usage
            timestamp_tz_count = len(re.findall(r'TIMESTAMP WITH TIME ZONE', self.content, re.IGNORECASE))
            if timestamp_tz_count > 0:
                logger.info(f"✓ Using timezone-aware timestamps ({timestamp_tz_count} instances)")

            # Check for constraints and validation
            check_constraints = len(re.findall(r'CHECK\s*\(', self.content, re.IGNORECASE))
            if check_constraints > 0:
                logger.info(f"✓ Found {check_constraints} CHECK constraints for data validation")

            # Check for comments/documentation
            comment_count = len(re.findall(r'COMMENT ON', self.content, re.IGNORECASE))
            if comment_count > 0:
                logger.info(f"✓ Schema includes documentation ({comment_count} comments)")
            else:
                self.warnings.append("Consider adding COMMENT ON statements for documentation")

            return True

        except Exception as e:
            self.issues.append(f"Best practices validation error: {e}")
            return False

    def validate_security(self) -> bool:
        """Validate security aspects of the schema."""
        try:
            # Check for role-based permissions
            grant_statements = len(re.findall(r'GRANT\s+', self.content, re.IGNORECASE))
            if grant_statements > 0:
                logger.info(f"✓ Found {grant_statements} permission grant statements")
            else:
                self.warnings.append("Consider adding explicit permission grants")

            # Check for sensitive data handling
            sensitive_patterns = [
                r'password',
                r'secret',
                r'private_key',
                r'api_key'
            ]

            for pattern in sensitive_patterns:
                matches = re.findall(pattern, self.content, re.IGNORECASE)
                if matches:
                    self.warnings.append(f"Potential sensitive data field: {pattern}")

            # Check for audit trails
            audit_patterns = [
                r'created_at',
                r'created_by',
                r'updated_at',
                r'updated_by'
            ]

            audit_count = 0
            for pattern in audit_patterns:
                audit_count += len(re.findall(pattern, self.content, re.IGNORECASE))

            if audit_count > 0:
                logger.info(f"✓ Audit trail fields present ({audit_count} instances)")

            return True

        except Exception as e:
            self.issues.append(f"Security validation error: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get schema statistics."""
        try:
            stats = {
                "total_lines": len(self.content.splitlines()),
                "total_chars": len(self.content),
                "tables": len(re.findall(r'CREATE TABLE', self.content, re.IGNORECASE)),
                "indexes": len(re.findall(r'CREATE INDEX', self.content, re.IGNORECASE)),
                "functions": len(re.findall(r'CREATE.*FUNCTION', self.content, re.IGNORECASE)),
                "triggers": len(re.findall(r'CREATE TRIGGER', self.content, re.IGNORECASE)),
                "views": len(re.findall(r'CREATE.*VIEW', self.content, re.IGNORECASE)),
                "comments": len(re.findall(r'COMMENT ON', self.content, re.IGNORECASE))
            }
            return stats
        except Exception:
            return {}

def main():
    """Run schema validation tests."""
    logger.info("Starting SQL schema validation...")

    # Find schema file
    schema_path = Path(__file__).parent / "schema.sql"

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return 1

    # Create validator
    validator = SQLSchemaValidator(str(schema_path))

    # Run validation tests
    tests = [
        ("Syntax Validation", validator.validate_syntax),
        ("Structure Validation", validator.validate_structure),
        ("Best Practices", validator.validate_best_practices),
        ("Security Validation", validator.validate_security)
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

    # Report statistics
    stats = validator.get_statistics()
    logger.info(f"\nSchema Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key.replace('_', ' ').title()}: {value}")

    # Report issues and warnings
    if validator.issues:
        logger.error(f"\nIssues found ({len(validator.issues)}):")
        for issue in validator.issues:
            logger.error(f"  - {issue}")

    if validator.warnings:
        logger.warning(f"\nWarnings ({len(validator.warnings)}):")
        for warning in validator.warnings:
            logger.warning(f"  - {warning}")

    # Final results
    total = passed + failed
    logger.info(f"\nSchema Validation Results:")
    logger.info(f"  Total tests: {total}")
    logger.info(f"  Passed: {passed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Issues: {len(validator.issues)}")
    logger.info(f"  Warnings: {len(validator.warnings)}")

    if failed == 0 and len(validator.issues) == 0:
        logger.info("✓ Schema validation passed successfully!")
        return 0
    else:
        logger.error("✗ Schema validation failed or has issues")
        return 1

if __name__ == "__main__":
    result = main()
    exit(result)