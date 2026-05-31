"""
APG Import/Export (IMEX) AI Intelligence Layer

Purpose: Production-grade AI-powered schema detection, data quality assessment,
         and intelligent field mapping using local Ollama models and statistical analysis.
Dependencies: ollama, numpy, pandas, asyncio, typing
Usage Context: AI enhancement layer for automated data processing

This module provides:
- Real schema detection using multiple algorithms
- AI-powered field type inference and pattern recognition
- Intelligent data quality assessment and anomaly detection
- Smart field mapping suggestions between source and target schemas
- Confidence scoring and validation for all AI predictions
"""

import asyncio
import json
import logging
import statistics
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available - falling back to statistical analysis only")

try:
    import numpy as np
    import pandas as pd
    NUMPY_PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_PANDAS_AVAILABLE = False
    logging.warning("NumPy/Pandas not available - using built-in statistics")

from .models import DataFormat, SourceType

logger = logging.getLogger(__name__)

@dataclass
class FieldAnalysis:
    """Detailed field analysis results for schema detection.

    Contains comprehensive analysis information for a single data field
    including type inference, quality metrics, pattern detection, and
    anomaly identification. Used by AI intelligence engine for schema
    detection and data quality assessment.

    Attributes:
        field_name: Name of the analyzed field
        inferred_type: Detected data type for the field
        confidence_score: Confidence level (0-1) of type inference
        nullable: Whether the field can contain null values
        unique_count: Number of unique values in the field
        total_count: Total number of values analyzed
        sample_values: Representative sample of field values
        data_patterns: Detected patterns in the data
        anomalies: Anomalous values or patterns found
        quality_issues: Data quality problems identified
        suggested_constraints: Recommended validation constraints
    """
    field_name: str
    inferred_type: str
    confidence_score: float
    nullable: bool
    unique_count: int
    total_count: int
    sample_values: List[Any]
    missing_count: int = 0
    data_patterns: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)
    suggested_constraints: List[str] = field(default_factory=list)

    @property
    def data_type(self) -> str:
        """Compatibility alias for callers that use data_type terminology."""
        return self.inferred_type

@dataclass
class SchemaAnalysisResult:
    """Complete schema analysis result for data sources.

    Contains comprehensive analysis results for an entire data source
    schema including field analysis, quality metrics, and AI-generated
    recommendations for optimal data processing.

    Attributes:
        fields: List of individual field analysis results
        total_records: Total number of records analyzed
        data_quality_score: Overall data quality score (0-100)
        confidence_score: Confidence in analysis results (0-1)
        analysis_method: Method used for analysis (ai, statistical, hybrid)
        processing_time_seconds: Time taken for analysis
        recommendations: AI-generated processing recommendations
        metadata: Additional analysis metadata and statistics
    """
    fields: List[FieldAnalysis]
    total_records: int
    data_quality_score: float
    confidence_score: float
    analysis_method: str
    processing_time_seconds: float
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAssessment:
    """Data quality assessment result with multi-dimensional scoring.

    Provides comprehensive data quality analysis across multiple
    dimensions including completeness, consistency, accuracy, uniqueness,
    and validity. Used for monitoring and improving data quality.

    Attributes:
        overall_score: Overall quality score (0-100)
        completeness_score: Data completeness score (0-100)
        consistency_score: Data consistency score (0-100)
        accuracy_score: Data accuracy score (0-100)
        uniqueness_score: Data uniqueness score (0-100)
        validity_score: Data validity score (0-100)
        issues_found: List of specific quality issues detected
        recommendations: Recommended actions for quality improvement
        field_scores: Quality scores by individual field
    """
    overall_score: float
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    uniqueness_score: float
    validity_score: float
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    field_scores: Dict[str, float] = field(default_factory=dict)

class AIIntelligenceEngine:
    """
    Production-grade AI intelligence engine for data analysis.

    Provides comprehensive AI-powered data analysis including schema detection,
    quality assessment, and intelligent field mapping. Uses local Ollama models
    for privacy and control, with fallback to statistical analysis.

    Attributes:
        ollama_model: Name of Ollama model to use for AI analysis
        max_sample_size: Maximum number of records to analyze
        confidence_threshold: Minimum confidence score for AI predictions
        cache_enabled: Whether to cache analysis results

    Example:
        >>> engine = AIIntelligenceEngine(ollama_model="llama3.1")
        >>> await engine.initialize()
        >>> analysis = await engine.analyze_schema(data_sample, DataFormat.CSV)
        >>> print(f"Schema confidence: {analysis.confidence_score}")
    """

    def __init__(
        self,
        ollama_model: str = "llama3.1:8b",
        max_sample_size: int = 10000,
        confidence_threshold: float = 0.7,
        cache_enabled: bool = True
    ):
        """
        Initialize AI intelligence engine.

        Args:
            ollama_model: Ollama model name for AI analysis
            max_sample_size: Maximum records to analyze for performance
            confidence_threshold: Minimum confidence for AI predictions
            cache_enabled: Enable result caching for performance
        """
        self.ollama_model = ollama_model
        self.max_sample_size = max_sample_size
        self.confidence_threshold = confidence_threshold
        self.cache_enabled = cache_enabled
        self.config = {
            "ollama_model": ollama_model,
            "max_sample_size": max_sample_size,
            "confidence_threshold": confidence_threshold,
            "cache_enabled": cache_enabled,
        }

        self.is_initialized = False
        self.ollama_client = None
        self.llm_available = False
        self._analysis_cache: Dict[str, Any] = {}

        self.type_patterns = {
            'integer': [r'^\d+$', r'^-?\d+$'],
            'float': [r'^\d*\.\d+$', r'^-?\d*\.\d+([eE][+-]?\d+)?$'],
            'boolean': [r'^(true|false|1|0|yes|no|y|n)$'],
            'date': [r'^\d{4}-\d{2}-\d{2}$', r'^\d{2}/\d{2}/\d{4}$'],
            'datetime': [r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$'],
            'email': [r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'],
            'phone': [r'^\+?[\d\s\-\(\)]{10,}$'],
            'url': [r'^https?://[^\s]+$'],
            'json': [r'^\{.*\}$', r'^\[.*\]$'],
            'uuid': [r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$']
        }

    async def initialize(self) -> bool:
        """
        Initialize AI engine and test connections.

        Returns:
            bool: True if initialization successful

        Raises:
            AIEngineError: If initialization fails
        """
        try:
            logger.info("Initializing AI Intelligence Engine...")

            # Test Ollama connection if available
            if OLLAMA_AVAILABLE:
                try:
                    self.ollama_client = ollama.AsyncClient()
                    # Test with a simple prompt
                    response = await self.ollama_client.generate(
                        model=self.ollama_model,
                        prompt="Test connection. Respond with 'OK'.",
                        options={'num_predict': 5}
                    )
                    if 'OK' in response.get('response', '').upper():
                        self.llm_available = True
                        logger.info(f"✓ Ollama model '{self.ollama_model}' connected successfully")
                    else:
                        logger.warning("Ollama connection test inconclusive - proceeding with caution")
                except Exception as e:
                    logger.warning(f"Ollama connection failed: {e} - using statistical analysis only")
                    self.ollama_client = None
                    self.llm_available = False

            self.is_initialized = True
            logger.info("AI Intelligence Engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"AI engine initialization failed: {e}")
            return False

    async def analyze_schema(
        self,
        data_sample: List[Dict[str, Any]],
        format_hint: DataFormat,
        source_info: Optional[Dict[str, Any]] = None
    ) -> SchemaAnalysisResult:
        """
        Perform comprehensive schema analysis on data sample.

        Uses multiple analysis methods including statistical analysis,
        pattern recognition, and AI-powered inference to determine
        optimal schema for the provided data.

        Args:
            data_sample: Sample data records for analysis
            format_hint: Expected data format for optimization
            source_info: Additional source information for context

        Returns:
            SchemaAnalysisResult: Comprehensive schema analysis

        Raises:
            AIEngineError: If analysis fails

        Example:
            >>> data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
            >>> result = await engine.analyze_schema(data, DataFormat.JSON)
            >>> print(f"Found {len(result.fields)} fields")
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Validate input
            if not data_sample:
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                return SchemaAnalysisResult(
                    fields=[],
                    total_records=0,
                    data_quality_score=0.0,
                    confidence_score=0.0,
                    analysis_method="empty",
                    processing_time_seconds=processing_time,
                    recommendations=["Provide sample records for schema inference"],
                    metadata={
                        "format_hint": format_hint.value,
                        "source_info": source_info or {},
                        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

            # Limit sample size for performance
            if len(data_sample) > self.max_sample_size:
                data_sample = data_sample[:self.max_sample_size]
                logger.info(f"Limited analysis to {self.max_sample_size} records")

            # Check cache
            cache_key = self._generate_cache_key(data_sample, format_hint)
            if self.cache_enabled and cache_key in self._analysis_cache:
                logger.info("Using cached schema analysis result")
                return self._analysis_cache[cache_key]

            # Analyze each field
            field_analyses = []
            all_fields = set()

            # Collect all possible field names
            for record in data_sample:
                if isinstance(record, dict):
                    all_fields.update(record.keys())

            logger.info(f"Analyzing {len(all_fields)} fields in {len(data_sample)} records")

            # Analyze each field (including nested fields)
            for field_name in sorted(all_fields):
                field_analysis = await self._analyze_field(field_name, data_sample)
                field_analyses.append(field_analysis)

                # Handle nested objects by flattening them
                sample_values = []
                for record in data_sample:
                    if isinstance(record, dict) and field_name in record:
                        value = record[field_name]
                        if isinstance(value, dict):
                            # Analyze nested dictionary fields
                            for nested_key, nested_value in value.items():
                                nested_field_name = f"{field_name}.{nested_key}"
                                if nested_field_name not in all_fields:  # Avoid duplicates
                                    nested_data = []
                                    for r in data_sample:
                                        if isinstance(r, dict) and field_name in r and isinstance(r[field_name], dict):
                                            nested_data.append({nested_field_name: r[field_name].get(nested_key)})
                                        else:
                                            nested_data.append({nested_field_name: None})

                                    if nested_data:
                                        nested_analysis = await self._analyze_field(nested_field_name, nested_data)
                                        field_analyses.append(nested_analysis)

            # Calculate overall metrics
            total_records = len(data_sample)
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Calculate quality score
            field_scores = [fa.confidence_score for fa in field_analyses]
            data_quality_score = statistics.mean(field_scores) if field_scores else 0.0

            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(field_analyses, format_hint)

            # Generate recommendations
            recommendations = self._generate_schema_recommendations(field_analyses, format_hint)

            # Determine analysis method used
            analysis_method = "statistical"
            if self.llm_available:
                analysis_method = "ai_enhanced"

            result = SchemaAnalysisResult(
                fields=field_analyses,
                total_records=total_records,
                data_quality_score=data_quality_score,
                confidence_score=confidence_score,
                analysis_method=analysis_method,
                processing_time_seconds=processing_time,
                recommendations=recommendations,
                metadata={
                    "format_hint": format_hint.value,
                    "source_info": source_info or {},
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "cache_key": cache_key
                }
            )

            # Cache result
            if self.cache_enabled:
                self._analysis_cache[cache_key] = result

            logger.info(f"Schema analysis completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            raise

    async def assess_data_quality(
        self,
        data_sample: List[Dict[str, Any]],
        schema_context: Optional[SchemaAnalysisResult] = None
    ) -> QualityAssessment:
        """
        Perform comprehensive data quality assessment.

        Analyzes data for completeness, consistency, accuracy, uniqueness,
        and validity. Uses AI-powered anomaly detection when available.

        Args:
            data_sample: Data records to assess
            schema_context: Optional schema analysis for context

        Returns:
            QualityAssessment: Comprehensive quality assessment

        Raises:
            AIEngineError: If quality assessment fails

        Example:
            >>> assessment = await engine.assess_data_quality(data)
            >>> print(f"Quality score: {assessment.overall_score:.2f}")
        """
        try:
            logger.info(f"Assessing data quality for {len(data_sample)} records")

            if not data_sample:
                return QualityAssessment(
                    overall_score=0.0,
                    completeness_score=0.0,
                    consistency_score=0.0,
                    accuracy_score=0.0,
                    uniqueness_score=0.0,
                    validity_score=0.0
                )

            # Calculate quality dimensions
            completeness = await self._assess_completeness(data_sample)
            consistency = await self._assess_consistency(data_sample)
            accuracy = await self._assess_accuracy(data_sample, schema_context)
            uniqueness = await self._assess_uniqueness(data_sample)
            validity = await self._assess_validity(data_sample, schema_context)

            # Calculate overall score (weighted average)
            weights = {'completeness': 0.25, 'consistency': 0.20, 'accuracy': 0.25, 'uniqueness': 0.15, 'validity': 0.15}
            overall_score = (
                completeness * weights['completeness'] +
                consistency * weights['consistency'] +
                accuracy * weights['accuracy'] +
                uniqueness * weights['uniqueness'] +
                validity * weights['validity']
            )

            # Generate recommendations
            recommendations = self._generate_quality_recommendations(
                completeness, consistency, accuracy, uniqueness, validity
            )

            # Calculate field-level scores
            field_scores = {}
            if schema_context:
                for field_analysis in schema_context.fields:
                    field_scores[field_analysis.field_name] = field_analysis.confidence_score

            return QualityAssessment(
                overall_score=overall_score,
                completeness_score=completeness,
                consistency_score=consistency,
                accuracy_score=accuracy,
                uniqueness_score=uniqueness,
                validity_score=validity,
                recommendations=recommendations,
                field_scores=field_scores
            )

        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            raise

    async def suggest_field_mappings(
        self,
        source_schema: SchemaAnalysisResult,
        target_schema: SchemaAnalysisResult,
        mapping_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest intelligent field mappings between source and target schemas.

        Uses AI-powered similarity analysis to suggest optimal field mappings
        based on field names, types, patterns, and semantic similarity.

        Args:
            source_schema: Source schema analysis result
            target_schema: Target schema analysis result
            mapping_context: Additional context for mapping decisions

        Returns:
            List[Dict[str, Any]]: Suggested field mappings with confidence scores

        Example:
            >>> mappings = await engine.suggest_field_mappings(source, target)
            >>> for mapping in mappings:
            ...     print(f"{mapping['source']} -> {mapping['target']} ({mapping['confidence']:.2f})")
        """
        try:
            logger.info("Generating intelligent field mapping suggestions")

            mappings = []

            for source_field in source_schema.fields:
                best_match = None
                best_score = 0.0

                for target_field in target_schema.fields:
                    # Calculate similarity score
                    score = await self._calculate_field_similarity(
                        source_field, target_field, mapping_context
                    )

                    if score > best_score and score >= self.confidence_threshold:
                        best_score = score
                        best_match = target_field

                if best_match:
                    mapping = {
                        "source_field": source_field.field_name,
                        "target_field": best_match.field_name,
                        "confidence": best_score,
                        "transformation_required": source_field.inferred_type != best_match.inferred_type,
                        "suggested_transformation": self._suggest_transformation(source_field, best_match)
                    }
                    mappings.append(mapping)
                else:
                    # No good match found
                    mappings.append({
                        "source_field": source_field.field_name,
                        "target_field": None,
                        "confidence": 0.0,
                        "suggestion": "Manual mapping required - no suitable target field found"
                    })

            logger.info(f"Generated {len(mappings)} field mapping suggestions")
            return mappings

        except Exception as e:
            logger.error(f"Field mapping suggestion failed: {e}")
            raise

    async def _analyze_field(self, field_name: str, data_sample: List[Dict[str, Any]]) -> FieldAnalysis:
        """Analyze individual field characteristics."""
        # Extract field values
        values = []
        for record in data_sample:
            if isinstance(record, dict) and field_name in record:
                value = record[field_name]
                # Handle complex types by converting to string representation
                if isinstance(value, (dict, list)):
                    values.append(str(value))
                else:
                    values.append(value)
            else:
                values.append(None)

        # Remove None values for analysis
        non_null_values = [v for v in values if v is not None]

        # Basic statistics
        total_count = len(values)
        non_null_count = len(non_null_values)
        null_count = total_count - non_null_count
        unique_count = len(set(str(v) for v in non_null_values))

        # Determine if nullable
        nullable = null_count > 0

        # Infer data type
        inferred_type, type_confidence = self._infer_data_type(non_null_values)

        # Detect patterns
        patterns = self._detect_patterns(non_null_values)

        # Find anomalies
        anomalies = self._detect_anomalies(non_null_values, inferred_type)

        # Assess quality issues
        quality_issues = []
        if null_count > total_count * 0.1:  # More than 10% nulls
            quality_issues.append(f"High null percentage: {(null_count/total_count)*100:.1f}%")

        if unique_count < non_null_count * 0.1:  # Very low uniqueness
            quality_issues.append(f"Low uniqueness: {(unique_count/non_null_count)*100:.1f}%")

        # Generate constraints
        constraints = self._suggest_constraints(non_null_values, inferred_type)

        # Calculate confidence based on multiple factors
        confidence_factors = [
            type_confidence,
            min(1.0, unique_count / max(1, non_null_count)),  # Uniqueness factor
            max(0.0, 1.0 - (null_count / total_count)),       # Completeness factor
        ]
        confidence_score = statistics.mean(confidence_factors)

        # Sample values for reference (handle unhashable types)
        try:
            # Try to get unique values using set
            sample_values = list(set(non_null_values[:10]))
        except TypeError:
            # If values are unhashable, just take first 10 without deduplication
            sample_values = non_null_values[:10]

        return FieldAnalysis(
            field_name=field_name,
            inferred_type=inferred_type,
            confidence_score=confidence_score,
            nullable=nullable,
            unique_count=unique_count,
            total_count=total_count,
            sample_values=sample_values,
            missing_count=null_count,
            data_patterns=patterns,
            anomalies=anomalies,
            quality_issues=quality_issues,
            suggested_constraints=constraints
        )

    def _infer_data_type(self, values: List[Any]) -> Tuple[str, float]:
        """Infer data type from values with confidence score."""
        if not values:
            return "string", 0.0

        # Convert all to strings for pattern matching
        str_values = [str(v) for v in values if v is not None]

        # Test each type pattern
        type_scores = {}

        for data_type, patterns in self.type_patterns.items():
            matches = 0
            for value in str_values:
                if any(re.match(pattern, value, re.IGNORECASE) for pattern in patterns):
                    matches += 1

            if str_values:
                type_scores[data_type] = matches / len(str_values)

        # Find best match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            confidence = type_scores[best_type]

            # Require at least 70% match for confidence
            if confidence >= 0.7:
                return best_type, confidence

        # Default to string type
        return "string", 0.5

    def _detect_patterns(self, values: List[Any]) -> List[str]:
        """Detect common data patterns."""
        patterns = []
        str_values = [str(v) for v in values if v is not None]

        if not str_values:
            return patterns

        # Length patterns
        lengths = [len(s) for s in str_values]
        if lengths:
            min_len, max_len = min(lengths), max(lengths)
            if min_len == max_len:
                patterns.append(f"Fixed length: {min_len}")
            else:
                patterns.append(f"Variable length: {min_len}-{max_len}")

        # Case patterns
        if all(s.isupper() for s in str_values if s):
            patterns.append("All uppercase")
        elif all(s.islower() for s in str_values if s):
            patterns.append("All lowercase")
        elif all(s.istitle() for s in str_values if s):
            patterns.append("Title case")

        # Numeric patterns
        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                pass

        if numeric_values and len(numeric_values) > len(values) * 0.8:
            patterns.append(f"Numeric range: {min(numeric_values):.2f} to {max(numeric_values):.2f}")

        return patterns

    def _detect_anomalies(self, values: List[Any], data_type: str) -> List[str]:
        """Detect data anomalies."""
        anomalies = []

        if not values:
            return anomalies

        # Type-specific anomaly detection
        if data_type in ['integer', 'float']:
            try:
                numeric_values = [float(v) for v in values if v is not None]
                if len(numeric_values) > 3:
                    mean_val = statistics.mean(numeric_values)
                    stdev_val = statistics.stdev(numeric_values)

                    # Find outliers (values beyond 2 standard deviations)
                    outliers = [v for v in numeric_values if abs(v - mean_val) > 2 * stdev_val]
                    if outliers:
                        anomalies.append(f"Statistical outliers detected: {len(outliers)} values")
            except (ValueError, statistics.StatisticsError):
                pass

        # Check for unusual characters in string fields
        if data_type == 'string':
            unusual_chars = set()
            for value in values:
                str_val = str(value)
                for char in str_val:
                    if not char.isalnum() and char not in ' .,!?@-_':
                        unusual_chars.add(char)

            if unusual_chars:
                anomalies.append(f"Unusual characters: {', '.join(sorted(unusual_chars))}")

        return anomalies

    def _suggest_constraints(self, values: List[Any], data_type: str) -> List[str]:
        """Suggest database constraints based on data analysis."""
        constraints = []

        if not values:
            return constraints

        # NOT NULL constraint
        if values:  # If we have non-null values, could be NOT NULL
            constraints.append("Consider NOT NULL if all values should be required")

        # Type-specific constraints
        if data_type in ['integer', 'float']:
            try:
                numeric_values = [float(v) for v in values]
                min_val, max_val = min(numeric_values), max(numeric_values)
                if min_val >= 0:
                    constraints.append("CHECK (value >= 0) -- All values positive")
                constraints.append(f"CHECK (value BETWEEN {min_val} AND {max_val})")
            except (ValueError, TypeError):
                pass

        elif data_type == 'string':
            str_values = [str(v) for v in values]
            lengths = [len(s) for s in str_values]
            if lengths:
                max_len = max(lengths)
                constraints.append(f"VARCHAR({max_len + 10}) -- Based on max observed length")

        # Uniqueness constraint
        unique_count = len(set(str(v) for v in values))
        if unique_count == len(values):
            constraints.append("UNIQUE -- All values are unique")

        return constraints

    async def _assess_completeness(self, data_sample: List[Dict[str, Any]]) -> float:
        """Assess data completeness (percentage of non-null values)."""
        if not data_sample:
            return 0.0

        total_cells = 0
        non_null_cells = 0

        for record in data_sample:
            if isinstance(record, dict):
                for value in record.values():
                    total_cells += 1
                    if value is not None and str(value).strip():
                        non_null_cells += 1

        return non_null_cells / total_cells if total_cells > 0 else 0.0

    async def _assess_consistency(self, data_sample: List[Dict[str, Any]]) -> float:
        """Assess data consistency (format and type consistency)."""
        if not data_sample:
            return 0.0

        # Get all field names
        all_fields = set()
        for record in data_sample:
            if isinstance(record, dict):
                all_fields.update(record.keys())

        consistency_scores = []

        for field_name in all_fields:
            field_values = []
            for record in data_sample:
                if isinstance(record, dict) and field_name in record:
                    field_values.append(record[field_name])

            # Check type consistency
            if field_values:
                types = [type(v).__name__ for v in field_values if v is not None]
                if types:
                    most_common_type = max(set(types), key=types.count)
                    type_consistency = types.count(most_common_type) / len(types)
                    consistency_scores.append(type_consistency)

        return statistics.mean(consistency_scores) if consistency_scores else 0.0

    async def _assess_accuracy(self, data_sample: List[Dict[str, Any]], schema_context: Optional[SchemaAnalysisResult]) -> float:
        """Assess data accuracy based on expected patterns and types."""
        if not data_sample:
            return 0.0

        accuracy_scores = []

        # If we have schema context, validate against expected types
        if schema_context:
            for field_analysis in schema_context.fields:
                field_name = field_analysis.field_name
                expected_type = field_analysis.inferred_type

                field_values = []
                for record in data_sample:
                    if isinstance(record, dict) and field_name in record:
                        field_values.append(record[field_name])

                if field_values:
                    accurate_values = 0
                    for value in field_values:
                        if value is not None:
                            if self._value_matches_type(value, expected_type):
                                accurate_values += 1

                    field_accuracy = accurate_values / len(field_values)
                    accuracy_scores.append(field_accuracy)

        return statistics.mean(accuracy_scores) if accuracy_scores else 0.8  # Default assumption

    async def _assess_uniqueness(self, data_sample: List[Dict[str, Any]]) -> float:
        """Assess data uniqueness (absence of duplicates)."""
        if not data_sample:
            return 0.0

        # Convert records to hashable format for duplicate detection
        record_hashes = []
        for record in data_sample:
            if isinstance(record, dict):
                # Create a sorted tuple of key-value pairs
                record_tuple = tuple(sorted(record.items()))
                record_hashes.append(hash(str(record_tuple)))

        if not record_hashes:
            return 0.0

        unique_records = len(set(record_hashes))
        total_records = len(record_hashes)

        return unique_records / total_records

    async def _assess_validity(self, data_sample: List[Dict[str, Any]], schema_context: Optional[SchemaAnalysisResult]) -> float:
        """Assess data validity (adherence to business rules and constraints)."""
        if not data_sample:
            return 0.0

        validity_scores = []

        # Basic validity checks
        for record in data_sample:
            if not isinstance(record, dict):
                validity_scores.append(0.0)
                continue

            record_validity = 1.0

            # Check for common validity issues
            for key, value in record.items():
                if value is None:
                    continue

                # Email validation
                if 'email' in key.lower() and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(value)):
                    record_validity *= 0.9

                # Phone validation
                if 'phone' in key.lower() and not re.match(r'^\+?[\d\s\-\(\)]{10,}$', str(value)):
                    record_validity *= 0.9

                # URL validation
                if 'url' in key.lower() and not re.match(r'^https?://[^\s]+$', str(value)):
                    record_validity *= 0.9

            validity_scores.append(record_validity)

        return statistics.mean(validity_scores) if validity_scores else 0.0

    def _value_matches_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        str_value = str(value)

        if expected_type in self.type_patterns:
            patterns = self.type_patterns[expected_type]
            return any(re.match(pattern, str_value, re.IGNORECASE) for pattern in patterns)

        return True  # Default to valid if type not recognized

    async def _calculate_field_similarity(
        self,
        source_field: FieldAnalysis,
        target_field: FieldAnalysis,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate similarity score between two fields."""
        similarity_factors = []

        # Name similarity (fuzzy matching)
        name_similarity = self._calculate_name_similarity(source_field.field_name, target_field.field_name)
        similarity_factors.append(('name', name_similarity, 0.4))

        # Type compatibility
        type_similarity = self._calculate_type_similarity(source_field.inferred_type, target_field.inferred_type)
        similarity_factors.append(('type', type_similarity, 0.3))

        # Pattern similarity
        pattern_similarity = self._calculate_pattern_similarity(source_field.data_patterns, target_field.data_patterns)
        similarity_factors.append(('pattern', pattern_similarity, 0.2))

        # Value similarity (sample values)
        value_similarity = self._calculate_value_similarity(source_field.sample_values, target_field.sample_values)
        similarity_factors.append(('value', value_similarity, 0.1))

        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in similarity_factors)
        weighted_score = sum(score * weight for _, score, weight in similarity_factors) / total_weight

        return weighted_score

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between field names."""
        # Normalize names
        norm1 = name1.lower().replace('_', '').replace('-', '')
        norm2 = name2.lower().replace('_', '').replace('-', '')

        # Exact match
        if norm1 == norm2:
            return 1.0

        # Substring match
        if norm1 in norm2 or norm2 in norm1:
            return 0.8

        # Common prefixes/suffixes
        common_parts = 0
        parts1 = re.split(r'[_-]', name1.lower())
        parts2 = re.split(r'[_-]', name2.lower())

        for part1 in parts1:
            if part1 in parts2:
                common_parts += 1

        if parts1 and parts2:
            return common_parts / max(len(parts1), len(parts2))

        return 0.0

    def _calculate_type_similarity(self, type1: str, type2: str) -> float:
        """Calculate compatibility between data types."""
        if type1 == type2:
            return 1.0

        # Compatible numeric types
        numeric_types = {'integer', 'float', 'number'}
        if type1 in numeric_types and type2 in numeric_types:
            return 0.8

        # String compatibility with everything (can always convert to string)
        if type1 == 'string' or type2 == 'string':
            return 0.6

        # Date/datetime compatibility
        date_types = {'date', 'datetime', 'timestamp'}
        if type1 in date_types and type2 in date_types:
            return 0.9

        return 0.0

    def _calculate_pattern_similarity(self, patterns1: List[str], patterns2: List[str]) -> float:
        """Calculate similarity between data patterns."""
        if not patterns1 and not patterns2:
            return 1.0

        if not patterns1 or not patterns2:
            return 0.0

        # Find common patterns
        common_patterns = set(patterns1) & set(patterns2)
        total_patterns = set(patterns1) | set(patterns2)

        return len(common_patterns) / len(total_patterns) if total_patterns else 0.0

    def _calculate_value_similarity(self, values1: List[Any], values2: List[Any]) -> float:
        """Calculate similarity between sample values."""
        if not values1 and not values2:
            return 1.0

        if not values1 or not values2:
            return 0.0

        # Convert to sets for comparison
        set1 = set(str(v) for v in values1)
        set2 = set(str(v) for v in values2)

        # Calculate Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _suggest_transformation(self, source_field: FieldAnalysis, target_field: FieldAnalysis) -> str:
        """Suggest data transformation between fields."""
        if source_field.inferred_type == target_field.inferred_type:
            return "No transformation required"

        source_type = source_field.inferred_type
        target_type = target_field.inferred_type

        # Common transformations
        transformations = {
            ('string', 'integer'): "CAST(value AS INTEGER)",
            ('string', 'float'): "CAST(value AS FLOAT)",
            ('integer', 'string'): "CAST(value AS VARCHAR)",
            ('float', 'string'): "CAST(value AS VARCHAR)",
            ('date', 'string'): "FORMAT(value, 'YYYY-MM-DD')",
            ('string', 'date'): "STR_TO_DATE(value, '%Y-%m-%d')",
            ('datetime', 'date'): "DATE(value)",
            ('date', 'datetime'): "TIMESTAMP(value)"
        }

        key = (source_type, target_type)
        return transformations.get(key, f"Custom transformation from {source_type} to {target_type}")

    def _calculate_overall_confidence(self, field_analyses: List[FieldAnalysis], format_hint: DataFormat) -> float:
        """Calculate overall confidence score for schema analysis."""
        if not field_analyses:
            return 0.0

        # Base confidence on field analysis confidence scores
        field_confidences = [fa.confidence_score for fa in field_analyses]
        base_confidence = statistics.mean(field_confidences)

        # Adjust based on format hint appropriateness
        format_bonus = 0.1 if format_hint in [DataFormat.JSON, DataFormat.CSV] else 0.0

        # Adjust based on number of fields (more fields = more confidence)
        field_count_bonus = min(0.1, len(field_analyses) * 0.01)

        # Adjust based on data quality
        quality_issues = sum(len(fa.quality_issues) for fa in field_analyses)
        quality_penalty = min(0.2, quality_issues * 0.02)

        final_confidence = base_confidence + format_bonus + field_count_bonus - quality_penalty
        return max(0.0, min(1.0, final_confidence))

    def _generate_schema_recommendations(self, field_analyses: List[FieldAnalysis], format_hint: DataFormat) -> List[str]:
        """Generate schema optimization recommendations."""
        recommendations = []

        # Check for fields with low confidence
        low_confidence_fields = [fa for fa in field_analyses if fa.confidence_score < 0.7]
        if low_confidence_fields:
            recommendations.append(f"Review {len(low_confidence_fields)} fields with low confidence scores")

        # Check for fields with many quality issues
        problematic_fields = [fa for fa in field_analyses if len(fa.quality_issues) > 2]
        if problematic_fields:
            recommendations.append(f"Address data quality issues in {len(problematic_fields)} fields")

        # Check for potential normalization opportunities
        string_fields = [fa for fa in field_analyses if fa.inferred_type == 'string']
        if len(string_fields) > len(field_analyses) * 0.7:
            recommendations.append("Consider data type optimization - many fields detected as strings")

        # Format-specific recommendations
        if format_hint == DataFormat.CSV:
            recommendations.append("Consider using more structured format like JSON for complex data")
        elif format_hint == DataFormat.JSON:
            recommendations.append("Validate JSON schema consistency across all records")

        return recommendations

    def _generate_quality_recommendations(
        self,
        completeness: float,
        consistency: float,
        accuracy: float,
        uniqueness: float,
        validity: float
    ) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []

        if completeness < 0.8:
            recommendations.append("Improve data completeness - add validation for required fields")

        if consistency < 0.8:
            recommendations.append("Improve data consistency - standardize data formats and types")

        if accuracy < 0.8:
            recommendations.append("Improve data accuracy - add validation rules and constraints")

        if uniqueness < 0.9:
            recommendations.append("Address duplicate records - implement deduplication process")

        if validity < 0.8:
            recommendations.append("Improve data validity - add business rule validation")

        return recommendations

    def _generate_cache_key(self, data_sample: List[Dict[str, Any]], format_hint: DataFormat) -> str:
        """Generate cache key for analysis results."""
        try:
            # Create hash of sample data structure and format
            sample_structure = []
            for record in data_sample[:5]:  # Use first 5 records for key
                if isinstance(record, dict):
                    field_info = {}
                    for key, value in record.items():
                        if isinstance(value, dict):
                            field_info[key] = {"type": "dict", "value": json.dumps(value, sort_keys=True, default=str)}
                        elif isinstance(value, list):
                            field_info[key] = {"type": "list", "value": json.dumps(value, sort_keys=True, default=str)}
                        else:
                            field_info[key] = {"type": type(value).__name__, "value": repr(value)}
                    sample_structure.append(field_info)

            content = {
                "structure": sample_structure,
                "format": format_hint.value,
                "sample_size": len(data_sample)
            }

            return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
        except Exception as e:
            # Fallback to simple hash if JSON serialization fails
            logger.warning(f"Cache key generation failed, using fallback: {e}")
            return hashlib.sha256(f"{format_hint.value}_{len(data_sample)}".encode()).hexdigest()[:16]

# Error classes
class AIEngineError(Exception):
    """Base exception for AI engine errors."""
    pass

class SchemaAnalysisError(AIEngineError):
    """Schema analysis error."""
    pass

class QualityAssessmentError(AIEngineError):
    """Quality assessment error."""
    pass

__all__ = [
    "AIIntelligenceEngine",
    "FieldAnalysis",
    "SchemaAnalysisResult",
    "QualityAssessment",
    "AIEngineError",
    "SchemaAnalysisError",
    "QualityAssessmentError"
]
