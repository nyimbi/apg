#!/usr/bin/env python3
"""
APG ETLP Visual Field Mapping Engine
Advanced visual field mapping and transformation interface

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

from .models import Pipeline, Transformation, DataSource


class FieldDataType(str, Enum):
    """Supported field data types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIMESTAMP = "timestamp"
    JSON = "json"
    ARRAY = "array"
    BINARY = "binary"
    UUID = "uuid"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"


class TransformationFunction(str, Enum):
    """Available transformation functions"""
    DIRECT_COPY = "direct_copy"
    TYPE_CONVERT = "type_convert"
    FORMAT_STRING = "format_string"
    CONCATENATE = "concatenate"
    SUBSTRING = "substring"
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    TRIM = "trim"
    REPLACE = "replace"
    REGEX_EXTRACT = "regex_extract"
    DATE_FORMAT = "date_format"
    MATH_OPERATION = "math_operation"
    CONDITIONAL = "conditional"
    LOOKUP = "lookup"
    DEFAULT_VALUE = "default_value"
    CUSTOM_FUNCTION = "custom_function"
    AGGREGATE = "aggregate"
    SPLIT_FIELD = "split_field"
    JSON_EXTRACT = "json_extract"
    ENCRYPT = "encrypt"
    HASH = "hash"


class FieldSchema(BaseModel):
    """Schema definition for a database field"""
    
    model_config = ConfigDict(
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True
    )
    
    name: str = Field(..., description="Field name")
    data_type: FieldDataType = Field(..., description="Field data type")
    nullable: bool = Field(default=True, description="Can field be null")
    primary_key: bool = Field(default=False, description="Is primary key")
    auto_increment: bool = Field(default=False, description="Auto-incrementing field")
    max_length: Optional[int] = Field(None, description="Maximum field length")
    precision: Optional[int] = Field(None, description="Decimal precision")
    scale: Optional[int] = Field(None, description="Decimal scale")
    default_value: Optional[Any] = Field(None, description="Default field value")
    description: Optional[str] = Field(None, description="Field description")
    constraints: List[str] = Field(default_factory=list, description="Field constraints")
    foreign_key: Optional[str] = Field(None, description="Foreign key reference")
    indexed: bool = Field(default=False, description="Has database index")
    
    # Metadata for visual interface
    position: Tuple[int, int] = Field(default=(0, 0), description="Visual position (x, y)")
    sample_values: List[Any] = Field(default_factory=list, description="Sample field values")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Field statistics")


class TableSchema(BaseModel):
    """Schema definition for a database table"""
    
    model_config = ConfigDict(
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True
    )
    
    id: str = Field(default_factory=uuid7str, description="Unique schema identifier")
    name: str = Field(..., description="Table name")
    database: str = Field(..., description="Database name")
    schema_name: Optional[str] = Field(None, description="Schema/namespace name")
    fields: List[FieldSchema] = Field(..., description="Table fields")
    primary_keys: List[str] = Field(default_factory=list, description="Primary key fields")
    indexes: List[Dict[str, Any]] = Field(default_factory=list, description="Table indexes")
    constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Table constraints")
    row_count: Optional[int] = Field(None, description="Approximate row count")
    table_size_mb: Optional[float] = Field(None, description="Table size in MB")
    last_updated: Optional[datetime] = Field(None, description="Last table update")
    
    # Visual interface metadata
    position: Tuple[int, int] = Field(default=(0, 0), description="Visual position")
    collapsed: bool = Field(default=False, description="Collapsed in UI")


class FieldMapping(BaseModel):
    """Mapping configuration between source and target fields"""
    
    model_config = ConfigDict(
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True
    )
    
    id: str = Field(default_factory=uuid7str, description="Unique mapping identifier")
    source_field: str = Field(..., description="Source field name")
    target_field: str = Field(..., description="Target field name")
    transformation: TransformationFunction = Field(
        default=TransformationFunction.DIRECT_COPY,
        description="Transformation to apply"
    )
    transformation_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Transformation configuration"
    )
    
    # Conditional mapping
    condition: Optional[str] = Field(None, description="Mapping condition (SQL-like)")
    priority: int = Field(default=0, description="Mapping priority")
    
    # Data quality and validation
    validation_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Field validation rules"
    )
    error_handling: str = Field(default="skip", description="Error handling strategy")
    
    # Visual connection metadata
    connection_points: Dict[str, Tuple[int, int]] = Field(
        default_factory=dict,
        description="Visual connection coordinates"
    )
    connection_style: Dict[str, str] = Field(
        default_factory=dict,
        description="Visual connection styling"
    )
    
    # Metadata
    created_by: str = Field(..., description="User who created mapping")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = Field(None, description="Mapping notes")


class MappingConfiguration(BaseModel):
    """Complete mapping configuration between source and target"""
    
    model_config = ConfigDict(
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True
    )
    
    id: str = Field(default_factory=uuid7str, description="Configuration identifier")
    name: str = Field(..., description="Configuration name")
    description: Optional[str] = Field(None, description="Configuration description")
    
    # Schema definitions
    source_schema: TableSchema = Field(..., description="Source table schema")
    target_schema: TableSchema = Field(..., description="Target table schema")
    
    # Field mappings
    field_mappings: List[FieldMapping] = Field(
        default_factory=list,
        description="Field mapping definitions"
    )
    
    # Global transformation settings
    batch_size: int = Field(default=1000, description="Processing batch size")
    parallel_execution: bool = Field(default=True, description="Enable parallel processing")
    preserve_order: bool = Field(default=False, description="Preserve record order")
    
    # Data quality settings
    validate_schema: bool = Field(default=True, description="Validate schema compatibility")
    validate_data: bool = Field(default=True, description="Validate data quality")
    error_threshold: float = Field(default=0.05, description="Error threshold (5%)")
    
    # Performance settings
    memory_limit_mb: int = Field(default=512, description="Memory limit in MB")
    timeout_minutes: int = Field(default=60, description="Execution timeout")
    checkpoint_interval: int = Field(default=10000, description="Checkpoint every N records")
    
    # Metadata
    tenant_id: str = Field(..., description="APG tenant identifier")
    created_by: str = Field(..., description="User who created configuration")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0", description="Configuration version")


class FieldMapperService:
    """Service for field mapping and schema management"""
    
    def __init__(self, tenant_id: str, user_id: str):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.intelligent_mapper = IntelligentMappingEngine()
        self.schema_analyzer = SchemaAnalyzer()
        self.transformation_engine = TransformationEngine()
    
    async def analyze_schema(self, data_source_id: str, table_name: str) -> TableSchema:
        """Analyze and extract schema from data source"""
        
        # Load data source configuration
        data_source = await self._load_data_source(data_source_id)
        
        # Connect and analyze schema
        schema_info = await self._extract_schema_info(data_source, table_name)
        
        # Get sample data and statistics
        sample_data = await self._get_sample_data(data_source, table_name, limit=100)
        field_stats = await self._calculate_field_statistics(sample_data)
        
        # Build schema object
        fields = []
        for field_info in schema_info.fields:
            field_schema = FieldSchema(
                name=field_info['name'],
                data_type=self._map_database_type(field_info['type']),
                nullable=field_info.get('nullable', True),
                primary_key=field_info.get('primary_key', False),
                max_length=field_info.get('max_length'),
                precision=field_info.get('precision'),
                scale=field_info.get('scale'),
                default_value=field_info.get('default'),
                description=field_info.get('comment'),
                sample_values=sample_data.get(field_info['name'], [])[:5],
                statistics=field_stats.get(field_info['name'], {})
            )
            fields.append(field_schema)
        
        return TableSchema(
            name=table_name,
            database=data_source.connection_string.split('/')[-1],
            schema_name=schema_info.get('schema'),
            fields=fields,
            primary_keys=schema_info.get('primary_keys', []),
            indexes=schema_info.get('indexes', []),
            constraints=schema_info.get('constraints', []),
            row_count=schema_info.get('row_count'),
            table_size_mb=schema_info.get('table_size_mb'),
            last_updated=schema_info.get('last_updated')
        )
    
    async def generate_intelligent_mappings(
        self, 
        source_schema: TableSchema, 
        target_schema: TableSchema
    ) -> List[FieldMapping]:
        """Generate intelligent field mappings with AI assistance"""
        
        mappings = []
        
        # Get AI-powered mapping suggestions
        suggestions = await self.intelligent_mapper.suggest_mappings(
            source_schema, target_schema
        )
        
        for suggestion in suggestions:
            # Create field mapping with intelligent defaults
            mapping = FieldMapping(
                source_field=suggestion['source_field'],
                target_field=suggestion['target_field'],
                transformation=suggestion['transformation'],
                transformation_config=suggestion.get('config', {}),
                created_by=self.user_id
            )
            
            # Add validation rules based on field types
            validation_rules = await self._generate_validation_rules(
                source_schema.fields,
                target_schema.fields,
                suggestion
            )
            mapping.validation_rules = validation_rules
            
            mappings.append(mapping)
        
        return mappings
    
    async def validate_mapping_configuration(self, config: MappingConfiguration) -> Dict[str, Any]:
        """Validate mapping configuration and detect potential issues"""
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        # Schema compatibility validation
        schema_issues = await self._validate_schema_compatibility(
            config.source_schema, 
            config.target_schema
        )
        validation_result['warnings'].extend(schema_issues)
        
        # Field mapping validation
        mapping_issues = await self._validate_field_mappings(config.field_mappings)
        validation_result['errors'].extend(mapping_issues.get('errors', []))
        validation_result['warnings'].extend(mapping_issues.get('warnings', []))
        
        # Performance validation
        performance_suggestions = await self._analyze_performance_implications(config)
        validation_result['suggestions'].extend(performance_suggestions)
        
        # Data quality validation
        quality_issues = await self._validate_data_quality_settings(config)
        validation_result['warnings'].extend(quality_issues)
        
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    async def execute_mapping(self, config: MappingConfiguration) -> str:
        """Execute the field mapping configuration"""
        
        # Create pipeline from mapping configuration
        pipeline_config = await self._convert_to_pipeline_config(config)
        
        # Create ETL pipeline
        from .service import ETLPService
        etlp_service = ETLPService(self.tenant_id, self.user_id)
        pipeline = await etlp_service.create_pipeline(pipeline_config)
        
        # Execute pipeline
        execution_id = await etlp_service.execute_pipeline(pipeline.id)
        
        return execution_id
    
    async def _convert_to_pipeline_config(self, config: MappingConfiguration) -> Dict[str, Any]:
        """Convert mapping configuration to pipeline configuration"""
        
        pipeline_config = {
            'name': f"Field Mapping: {config.source_schema.name} -> {config.target_schema.name}",
            'description': config.description or f"Automated field mapping configuration",
            'execution_mode': 'batch',
            'steps': [
                {
                    'type': 'extract',
                    'source_table': config.source_schema.name,
                    'source_database': config.source_schema.database
                },
                {
                    'type': 'transform',
                    'field_mappings': [
                        {
                            'source': mapping.source_field,
                            'target': mapping.target_field,
                            'transformation': mapping.transformation.value,
                            'config': mapping.transformation_config,
                            'validation': mapping.validation_rules
                        }
                        for mapping in config.field_mappings
                    ]
                },
                {
                    'type': 'load',
                    'target_table': config.target_schema.name,
                    'target_database': config.target_schema.database,
                    'batch_size': config.batch_size
                }
            ],
            'max_parallelism': 4 if config.parallel_execution else 1,
            'timeout_minutes': config.timeout_minutes,
            'ai_optimization_enabled': True
        }
        
        return pipeline_config


class IntelligentMappingEngine:
    """AI-powered intelligent field mapping engine"""
    
    def __init__(self):
        self.name_similarity_threshold = 0.8
        self.type_compatibility_matrix = self._build_type_compatibility_matrix()
    
    async def suggest_mappings(
        self, 
        source_schema: TableSchema, 
        target_schema: TableSchema
    ) -> List[Dict[str, Any]]:
        """Generate intelligent mapping suggestions"""
        
        suggestions = []
        
        # Create field name similarity matrix
        similarity_matrix = await self._calculate_name_similarities(
            source_schema.fields, target_schema.fields
        )
        
        # Find best matches
        used_target_fields = set()
        
        for source_field in source_schema.fields:
            best_match = await self._find_best_field_match(
                source_field, target_schema.fields, similarity_matrix, used_target_fields
            )
            
            if best_match:
                suggestion = await self._create_mapping_suggestion(
                    source_field, best_match['field'], best_match['confidence']
                )
                suggestions.append(suggestion)
                used_target_fields.add(best_match['field'].name)
        
        return suggestions
    
    async def _calculate_name_similarities(
        self, 
        source_fields: List[FieldSchema], 
        target_fields: List[FieldSchema]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate field name similarity matrix"""
        
        similarity_matrix = {}
        
        for source_field in source_fields:
            similarity_matrix[source_field.name] = {}
            
            for target_field in target_fields:
                similarity = await self._calculate_field_similarity(
                    source_field, target_field
                )
                similarity_matrix[source_field.name][target_field.name] = similarity
        
        return similarity_matrix
    
    async def _calculate_field_similarity(
        self, 
        source_field: FieldSchema, 
        target_field: FieldSchema
    ) -> float:
        """Calculate similarity score between two fields"""
        
        # Name similarity (fuzzy matching)
        name_similarity = await self._fuzzy_name_match(
            source_field.name, target_field.name
        )
        
        # Type compatibility
        type_compatibility = self._get_type_compatibility(
            source_field.data_type, target_field.data_type
        )
        
        # Description similarity (if available)
        desc_similarity = 0.0
        if source_field.description and target_field.description:
            desc_similarity = await self._text_similarity(
                source_field.description, target_field.description
            )
        
        # Sample data pattern similarity
        sample_similarity = await self._sample_data_similarity(
            source_field.sample_values, target_field.sample_values
        )
        
        # Weighted combination
        total_similarity = (
            name_similarity * 0.4 +
            type_compatibility * 0.3 +
            desc_similarity * 0.2 +
            sample_similarity * 0.1
        )
        
        return total_similarity
    
    async def _fuzzy_name_match(self, name1: str, name2: str) -> float:
        """Calculate fuzzy string matching score"""
        
        # Normalize names
        norm1 = self._normalize_field_name(name1)
        norm2 = self._normalize_field_name(name2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Levenshtein distance
        distance = self._levenshtein_distance(norm1, norm2)
        max_len = max(len(norm1), len(norm2))
        
        if max_len == 0:
            return 1.0
        
        similarity = 1.0 - (distance / max_len)
        
        # Bonus for common patterns
        if self._has_common_patterns(norm1, norm2):
            similarity += 0.1
        
        return min(similarity, 1.0)
    
    def _normalize_field_name(self, name: str) -> str:
        """Normalize field name for comparison"""
        # Convert to lowercase and remove common prefixes/suffixes
        normalized = name.lower()
        normalized = re.sub(r'^(src_|tgt_|source_|target_)', '', normalized)
        normalized = re.sub(r'(_id|_key|_code)$', '', normalized)
        normalized = re.sub(r'[_\s]+', '_', normalized)
        return normalized.strip('_')
    
    def _build_type_compatibility_matrix(self) -> Dict[FieldDataType, Dict[FieldDataType, float]]:
        """Build type compatibility matrix"""
        
        matrix = {}
        
        # Define compatibility scores (0.0 = incompatible, 1.0 = perfect match)
        compatibilities = {
            # String types
            FieldDataType.STRING: {
                FieldDataType.STRING: 1.0,
                FieldDataType.EMAIL: 0.9,
                FieldDataType.URL: 0.9,
                FieldDataType.PHONE: 0.9,
                FieldDataType.UUID: 0.8,
                FieldDataType.JSON: 0.7,
                FieldDataType.DATE: 0.6,
                FieldDataType.DATETIME: 0.6,
            },
            
            # Numeric types
            FieldDataType.INTEGER: {
                FieldDataType.INTEGER: 1.0,
                FieldDataType.FLOAT: 0.9,
                FieldDataType.DECIMAL: 0.9,
                FieldDataType.BOOLEAN: 0.7,
                FieldDataType.STRING: 0.6,
            },
            
            FieldDataType.FLOAT: {
                FieldDataType.FLOAT: 1.0,
                FieldDataType.DECIMAL: 0.9,
                FieldDataType.INTEGER: 0.8,
                FieldDataType.STRING: 0.6,
            },
            
            # Date types
            FieldDataType.DATE: {
                FieldDataType.DATE: 1.0,
                FieldDataType.DATETIME: 0.9,
                FieldDataType.TIMESTAMP: 0.9,
                FieldDataType.STRING: 0.7,
            },
            
            FieldDataType.DATETIME: {
                FieldDataType.DATETIME: 1.0,
                FieldDataType.TIMESTAMP: 0.9,
                FieldDataType.DATE: 0.8,
                FieldDataType.STRING: 0.7,
            },
            
            # Special types
            FieldDataType.BOOLEAN: {
                FieldDataType.BOOLEAN: 1.0,
                FieldDataType.INTEGER: 0.7,
                FieldDataType.STRING: 0.6,
            },
            
            FieldDataType.JSON: {
                FieldDataType.JSON: 1.0,
                FieldDataType.STRING: 0.8,
                FieldDataType.ARRAY: 0.7,
            },
        }
        
        # Build full matrix with defaults
        for source_type in FieldDataType:
            matrix[source_type] = {}
            for target_type in FieldDataType:
                if source_type in compatibilities and target_type in compatibilities[source_type]:
                    matrix[source_type][target_type] = compatibilities[source_type][target_type]
                elif source_type == target_type:
                    matrix[source_type][target_type] = 1.0
                else:
                    matrix[source_type][target_type] = 0.2  # Low compatibility default
        
        return matrix


class TransformationEngine:
    """Engine for applying field transformations"""
    
    def __init__(self):
        self.transformation_functions = self._register_transformation_functions()
    
    def _register_transformation_functions(self) -> Dict[TransformationFunction, callable]:
        """Register available transformation functions"""
        
        return {
            TransformationFunction.DIRECT_COPY: self._direct_copy,
            TransformationFunction.TYPE_CONVERT: self._type_convert,
            TransformationFunction.FORMAT_STRING: self._format_string,
            TransformationFunction.CONCATENATE: self._concatenate,
            TransformationFunction.SUBSTRING: self._substring,
            TransformationFunction.UPPERCASE: self._uppercase,
            TransformationFunction.LOWERCASE: self._lowercase,
            TransformationFunction.TRIM: self._trim,
            TransformationFunction.REPLACE: self._replace,
            TransformationFunction.REGEX_EXTRACT: self._regex_extract,
            TransformationFunction.DATE_FORMAT: self._date_format,
            TransformationFunction.MATH_OPERATION: self._math_operation,
            TransformationFunction.CONDITIONAL: self._conditional,
            TransformationFunction.LOOKUP: self._lookup,
            TransformationFunction.DEFAULT_VALUE: self._default_value,
            TransformationFunction.CUSTOM_FUNCTION: self._custom_function,
        }
    
    async def apply_transformation(
        self, 
        value: Any, 
        transformation: TransformationFunction,
        config: Dict[str, Any]
    ) -> Any:
        """Apply transformation function to value"""
        
        if transformation not in self.transformation_functions:
            raise ValueError(f"Unknown transformation: {transformation}")
        
        transform_func = self.transformation_functions[transformation]
        return await transform_func(value, config)
    
    async def _direct_copy(self, value: Any, config: Dict[str, Any]) -> Any:
        """Direct copy transformation"""
        return value
    
    async def _type_convert(self, value: Any, config: Dict[str, Any]) -> Any:
        """Type conversion transformation"""
        target_type = config.get('target_type', 'string')
        
        if target_type == 'string':
            return str(value) if value is not None else None
        elif target_type == 'integer':
            return int(float(str(value))) if value is not None else None
        elif target_type == 'float':
            return float(str(value)) if value is not None else None
        elif target_type == 'boolean':
            return bool(value) if value is not None else None
        else:
            return value
    
    async def _format_string(self, value: Any, config: Dict[str, Any]) -> str:
        """String formatting transformation"""
        format_string = config.get('format', '{}')
        return format_string.format(value) if value is not None else None
    
    async def _concatenate(self, values: List[Any], config: Dict[str, Any]) -> str:
        """Concatenation transformation"""
        separator = config.get('separator', '')
        return separator.join(str(v) for v in values if v is not None)
    
    async def _substring(self, value: Any, config: Dict[str, Any]) -> str:
        """Substring transformation"""
        start = config.get('start', 0)
        end = config.get('end', None)
        
        if value is None:
            return None
        
        str_value = str(value)
        return str_value[start:end]
    
    async def _uppercase(self, value: Any, config: Dict[str, Any]) -> str:
        """Uppercase transformation"""
        return str(value).upper() if value is not None else None
    
    async def _lowercase(self, value: Any, config: Dict[str, Any]) -> str:
        """Lowercase transformation"""
        return str(value).lower() if value is not None else None
    
    async def _trim(self, value: Any, config: Dict[str, Any]) -> str:
        """Trim whitespace transformation"""
        return str(value).strip() if value is not None else None


class SchemaAnalyzer:
    """Utility class for schema analysis and statistics"""
    
    async def analyze_field_compatibility(
        self, 
        source_field: FieldSchema, 
        target_field: FieldSchema
    ) -> Dict[str, Any]:
        """Analyze compatibility between two fields"""
        
        compatibility = {
            'compatible': True,
            'confidence': 1.0,
            'issues': [],
            'recommendations': []
        }
        
        # Type compatibility
        if source_field.data_type != target_field.data_type:
            type_compat = await self._check_type_compatibility(
                source_field.data_type, target_field.data_type
            )
            
            compatibility['confidence'] *= type_compat['compatibility_score']
            
            if type_compat['conversion_required']:
                compatibility['recommendations'].append({
                    'type': 'transformation',
                    'function': 'type_convert',
                    'config': {'target_type': target_field.data_type.value}
                })
        
        # Size compatibility
        if source_field.max_length and target_field.max_length:
            if source_field.max_length > target_field.max_length:
                compatibility['issues'].append({
                    'type': 'size_mismatch',
                    'message': f"Source field length ({source_field.max_length}) exceeds target ({target_field.max_length})"
                })
                compatibility['recommendations'].append({
                    'type': 'transformation',
                    'function': 'substring',
                    'config': {'start': 0, 'end': target_field.max_length}
                })
        
        # Nullability compatibility  
        if not source_field.nullable and target_field.nullable:
            compatibility['recommendations'].append({
                'type': 'validation',
                'rule': 'not_null_check'
            })
        
        return compatibility