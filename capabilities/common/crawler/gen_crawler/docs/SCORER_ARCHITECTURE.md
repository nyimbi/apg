# Lindela Scorer Package Architecture Documentation

## Overview

This document provides a comprehensive analysis of the Lindela scorer package architecture and its precise operation, including detailed information on how summaries are generated.

## Core Architecture

The scorer package is built on a **protocol-based, hierarchical architecture** designed for conflict monitoring in the Horn of Africa. It follows a sophisticated hybrid approach that combines multiple extraction methods for optimal performance.

## Key Components

### 1. Base Framework (`core/`)

- **BaseScorerProtocol**: Unified interface all scorers implement
- **ScorerFactory**: Dynamic creation and management of different scorer types
- **Unified data models**: Standardized `ScoringResult` with 270+ intelligence fields
- **Metrics system**: Performance tracking and health monitoring

### 2. Scorer Types

**EventExtractor**: LangChain-based LLM scorer with multi-provider support
**EnhancedMLScorer**: Machine learning-based conflict scoring
**RulesBasedScorer**: Fast pattern-based extraction without ML dependencies
**HierarchicalExtractor**: Progressive 4-tier extraction system
**Hybrid System**: 90% LLM reduction through intelligent optimization

### 3. Processing Methodology

The system uses **intelligent decision-making** for processing:

```
Text Input → Fast Analysis → Confidence Assessment → Selective LLM Use → Result Aggregation
```

**Decision Logic**:
- **High Confidence (85%+)**: Fast methods only (rules, NER, patterns)
- **Medium Confidence (65-85%)**: Selective LLM for specific complex fields
- **Low Confidence (<65%)**: Full LLM fallback for comprehensive analysis

### 4. Hierarchical Extraction

**4-Tier Progressive System**:
- **Foundational**: Basic entity extraction (locations, people, organizations)
- **Immediate**: Quick conflict classification and sentiment
- **Structural**: Relationship analysis and context understanding  
- **Predictive**: Advanced inference and future event prediction

**Extraction Modes**:
- **FAST**: Foundational + Immediate (real-time processing)
- **STANDARD**: + Structural (balanced analysis)
- **COMPREHENSIVE**: All tiers (maximum intelligence)

### 5. Performance Optimizations

**Hybrid Approach Achievements**:
- **90% reduction** in LLM API calls
- **10x faster** processing (sub-100ms per document)
- **95% cost reduction** compared to pure LLM approaches
- **Real-time capability** for operational use

**Optimization Components**:
- **HybridNERExtractor**: spaCy + custom patterns for entities
- **TextGraphExtractor**: Relationship and causal chain detection
- **FastKeywordExtractor**: Ultra-fast keyword/tag extraction
- **Multi-level caching**: Memory, disk, and database caching

### 6. Data Flow

**Input Processing**:
1. Text content enters through unified interface
2. Fast extractors (rules, NER) provide initial analysis
3. Confidence assessment determines next steps
4. Selective LLM calls for complex fields if needed
5. Results aggregated into standardized format

**Output Structure**:
- **Standard parameters**: conflict_score, confidence, severity_level, etc.
- **Custom parameters**: Domain-specific intelligence fields
- **Metadata**: Processing metrics, confidence scores, method used
- **Validation**: Type checking and data consistency verification

### 7. Intelligence Fields

**270+ Structured Fields** including:
- **Conflict Analysis**: Types, severity, escalation patterns
- **Entity Extraction**: People, places, organizations, weapons
- **Temporal Analysis**: Event timing, duration, patterns
- **Geospatial Data**: Locations, boundaries, movement patterns
- **Causal Analysis**: Root causes, contributing factors
- **Predictive Indicators**: Early warning signals, trend analysis

### 8. Integration Architecture

**Database Integration**: Direct PostgreSQL integration with batch processing
**Multi-LLM Support**: Ollama, OpenAI, Claude, DeepSeek with automatic fallback
**External Systems**: spaCy NLP, Gensim topic modeling, TextGraph relationships
**Monitoring**: Comprehensive metrics, health checks, progress tracking

## Precise Operation Flow

1. **Initialization**: Factory creates appropriate scorer based on requirements
2. **Fast Analysis**: Rules and NER provide initial extraction in <10ms
3. **Confidence Assessment**: System evaluates if LLM is needed
4. **Selective Processing**: Only complex fields go to LLM if needed
5. **Result Aggregation**: All extractions combined into unified structure
6. **Validation**: Type checking and consistency verification
7. **Output**: Standardized ScoringResult with intelligence fields

The system achieves **production-grade performance** by intelligently balancing speed, accuracy, and cost through its hybrid approach, making it suitable for real-time conflict monitoring operations.

## Summary Generation Architecture

### Summary Types and Structure

The Lindela scorer package implements **three main types of summaries**:

1. **`event_summary`**: Brief event description (200-300 characters, max 500)
2. **`summary_30_words`**: Concise summary (max 35 words, max 200 characters)  
3. **`summary_100_words`**: Detailed summary (max 110 words, max 600 characters)

### Primary Generation Method: LLM-Based Extraction

#### Structured LLM Processing
- **Implementation**: `EventExtractor` with dedicated `SUMMARIES` cluster
- **Schema**: `SummariesSchema` with validation for word/character limits
- **Process**: Uses structured LLM extraction with specialized prompts

#### LLM Configuration
```python
ExtractionPhase(
    phase=ExtractionPhase.SUMMARY,
    system_prompt="You are a summary expert. Create concise summaries of the event.",
    user_prompt_template="Create summaries for this article:\n\n{article_text}",
    dependencies=["CORE_EVENT", "HUMAN_IMPACT", "GEOGRAPHIC", "TEMPORAL"],
    output_fields=["summary_30_words", "summary_100_words"]
)
```

#### Validation Rules
```python
@field_validator('summary_30_words')
def validate_30_word_summary(cls, v):
    if v and len(v.split()) > 35:  # 15% flexibility
        raise ValueError("30-word summary should not exceed 35 words")

@field_validator('summary_100_words') 
def validate_100_word_summary(cls, v):
    if v and len(v.split()) > 110:  # 10% flexibility
        raise ValueError("100-word summary should not exceed 110 words")
```

### Fallback Methods

#### 1. Gensim-Based Summarization
```python
def _generate_summary(self, text: str) -> str:
    """Generate summary using Gensim."""
    try:
        summary = gensim.summarization.summarize(text, word_count=50)
        return summary if summary else self._extract_first_sentences(text, 3)
    except:
        return self._extract_first_sentences(text, 3)
```

#### 2. Rule-Based Fallback
- **Method**: Extracts first 3 sentences when other methods fail
- **Implementation**: Used in Gensim analyzer and hybrid scorer
- **Purpose**: Ensures summaries are always available

#### 3. Hybrid Approach
- **Combines**: Topic analysis with rule-based extraction
- **Enhancement**: Uses first sentence as basic summary when LLM enhancement needed
- **Optimization**: Part of the 90% LLM reduction strategy

### Generation Process Flow

#### 1. Hierarchical Processing
```
Core Event Data → Geographic/Temporal → Human Impact → Summary Generation
```

#### 2. Context-Aware Generation
- **Dependencies**: Summaries generated after core extraction phases
- **Context**: Uses previously extracted data to inform summary content
- **Consistency**: Ensures summaries align with extracted event details

#### 3. Multi-Method Pipeline
```
LLM Extraction → Validation → Gensim Fallback → Rule-Based Fallback → Output
```

### Integration in Scoring Process

#### Standard Inclusion
- **Part of comprehensive extraction**: Always included in full scoring process
- **Cluster processing**: Handled as dedicated `SUMMARIES` cluster
- **Performance weight**: `event_summary` has importance weight of 2.5

#### Batch Processing
- **Scalable**: Can process summaries for large datasets
- **Efficient**: Part of optimized hierarchical extraction
- **Resumable**: Supports checkpoint-based processing

### Performance Characteristics

#### Speed Optimization
- **Fast path**: Uses non-LLM methods when confidence is high
- **Selective LLM**: Only uses LLM for complex summarization when needed
- **Caching**: Benefits from standard LLM response caching

#### Quality Assurance
- **Multi-level validation**: Word count, character limits, content quality
- **Fallback guarantee**: Always produces a summary through fallback chain
- **Consistency checking**: Validates summary against extracted data

### Key Implementation Files

1. **`core/models.py`**: Summary schema and validation
2. **`implementations/event_extractor.py`**: Primary LLM-based generation
3. **`implementations/gensim_text_analyzer.py`**: Gensim fallback method
4. **`implementations/hybrid_scorer.py`**: Hybrid approach integration
5. **`implementations/ml_scorer.py`**: ML-enhanced summarization

## Summary Generation System Features

The summary generation system is **robust, multi-layered, and context-aware**, ensuring high-quality summaries are always produced through intelligent method selection and comprehensive fallback mechanisms.

### Key Benefits

1. **Reliability**: Multiple fallback mechanisms ensure summaries are always generated
2. **Consistency**: Context-aware generation aligned with extracted intelligence
3. **Flexibility**: Multiple summary types for different use cases
4. **Performance**: Optimized processing with intelligent LLM usage
5. **Quality**: Validation and quality assurance throughout the pipeline

### Integration with Crawler Systems

The scorer package integrates seamlessly with the crawler systems in the Lindela platform:

- **Content Processing**: Processes crawled content through the scoring pipeline
- **Real-time Analysis**: Provides near real-time intelligence extraction
- **Batch Operations**: Supports large-scale content processing
- **Database Integration**: Stores results in structured format for retrieval
- **API Integration**: Provides programmatic access to scoring capabilities

This comprehensive architecture enables the Lindela platform to provide sophisticated conflict monitoring capabilities with high performance, reliability, and accuracy.

## Author and Attribution

**Author**: Nyimbi Odero  
**Company**: Datacraft (www.datacraft.co.ke)  
**Date**: July 2025  
**Version**: 1.0  

This documentation is part of the Lindela enhanced packages system for conflict monitoring in the Horn of Africa.