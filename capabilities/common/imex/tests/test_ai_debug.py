#!/usr/bin/env python3
"""
Debug test for AI integration issues.
"""
import asyncio
import logging
import tempfile
import json
from pathlib import Path

from ai_intelligence import AIIntelligenceEngine
from models import DataFormat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_json_schema_detection():
    """Debug JSON schema detection issue."""
    try:
        # Create AI engine
        ai_engine = AIIntelligenceEngine()
        await ai_engine.initialize()

        # Create test JSON data with nested objects
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
                    "login_count": 28
                },
                "tags": ["standard"],
                "active": True
            }
        ]

        print(f"Analyzing data with {len(test_data)} records")

        # Test schema analysis
        analysis = await ai_engine.analyze_schema(test_data, DataFormat.JSON)

        print(f"✓ Analysis completed successfully!")
        print(f"  Fields detected: {len(analysis.fields)}")
        print(f"  Confidence: {analysis.confidence_score:.2f}")
        print(f"  Processing time: {analysis.processing_time_seconds:.3f}s")

        for field in analysis.fields[:10]:  # Show first 10 fields
            print(f"  - {field.field_name}: {field.inferred_type} (confidence: {field.confidence_score:.2f})")

        return True

    except Exception as e:
        print(f"✗ JSON schema detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_quality():
    """Debug data quality assessment issue."""
    try:
        # Create AI engine
        ai_engine = AIIntelligenceEngine()
        await ai_engine.initialize()

        # Create test data with quality issues
        test_data = [
            {"id": 1, "name": "John Doe", "email": "john@example.com", "age": 30, "score": 95.5},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "age": 25, "score": 87.2},
            {"id": 3, "name": "", "email": "invalid-email", "age": -5, "score": 999.9},
            {"id": 4, "name": "Bob Johnson", "email": "bob@example.com", "age": 35, "score": 78.1},
        ]

        print(f"Assessing quality for {len(test_data)} records")

        # Test quality assessment
        assessment = await ai_engine.assess_data_quality(test_data)

        print(f"✓ Quality assessment completed!")
        print(f"  Overall score: {assessment.overall_score:.2f}")
        print(f"  Completeness: {assessment.completeness_score:.2f}")
        print(f"  Consistency: {assessment.consistency_score:.2f}")
        print(f"  Accuracy: {assessment.accuracy_score:.2f}")
        print(f"  Uniqueness: {assessment.uniqueness_score:.2f}")
        print(f"  Validity: {assessment.validity_score:.2f}")
        print(f"  Recommendations: {len(assessment.recommendations)}")

        # Verify all scores are valid
        scores = [
            assessment.overall_score,
            assessment.completeness_score,
            assessment.consistency_score,
            assessment.accuracy_score,
            assessment.uniqueness_score,
            assessment.validity_score
        ]

        for i, score in enumerate(scores):
            if not (0 <= score <= 1):
                print(f"✗ Invalid score at index {i}: {score}")
                return False

        print("✓ All quality scores are valid (0-1 range)")
        return True

    except Exception as e:
        print(f"✗ Data quality assessment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run debug tests."""
    print("=== AI Integration Debug Tests ===\n")

    print("Test 1: JSON Schema Detection")
    test1_result = await test_json_schema_detection()

    print("\nTest 2: Data Quality Assessment")
    test2_result = await test_data_quality()

    print(f"\n=== Results ===")
    print(f"JSON Schema Detection: {'PASS' if test1_result else 'FAIL'}")
    print(f"Data Quality Assessment: {'PASS' if test2_result else 'FAIL'}")

    return 0 if (test1_result and test2_result) else 1

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(result)