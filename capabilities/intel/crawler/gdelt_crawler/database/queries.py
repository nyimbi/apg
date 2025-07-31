"""
Optimized Database Queries for GDELT Data Analysis
==================================================

High-performance database queries for GDELT data analysis, conflict monitoring,
and event intelligence with PostgreSQL optimization and ML integration support.

Key Features:
- **Conflict Analysis**: Specialized queries for conflict detection and monitoring
- **Event Intelligence**: Advanced event querying and correlation analysis
- **Geographic Analysis**: Location-based queries with spatial optimization
- **Temporal Analysis**: Time-series queries for trend analysis
- **ML Integration**: Queries optimized for ML Deep Scorer results
- **Performance Optimization**: Indexed queries with query plan optimization
- **Real-time Analytics**: Fast queries for real-time monitoring
- **Aggregation Queries**: Statistical aggregations for reporting

Query Categories:
- **Event Queries**: Core event search and retrieval
- **Conflict Queries**: Conflict-specific analysis and monitoring
- **Geographic Queries**: Location-based event analysis
- **Temporal Queries**: Time-based analysis and trends
- **ML Queries**: ML scoring and confidence analysis

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, timezone
import asyncpg
import logging
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    data: List[Dict[str, Any]]
    total_count: int
    execution_time_ms: float
    query_metadata: Dict[str, Any]


class GDELTQueryOptimizer:
    """
    Optimized database queries for GDELT data analysis and monitoring.
    
    Provides high-performance queries for conflict analysis, event intelligence,
    and real-time monitoring with PostgreSQL optimization.
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    # ==================== EVENT QUERIES ====================
    
    async def get_events_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000,
        offset: int = 0,
        event_types: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        min_confidence: float = 0.0
    ) -> QueryResult:
        """
        Get events within a date range with optional filtering.
        
        Args:
            start_date: Start date for events
            end_date: End date for events
            limit: Maximum number of results
            offset: Offset for pagination
            event_types: List of event types to filter by
            countries: List of countries to filter by
            min_confidence: Minimum confidence score
            
        Returns:
            QueryResult with events and metadata
        """
        filters = ["published_at BETWEEN $1 AND $2"]
        params = [start_date, end_date]
        param_count = 2
        
        if event_types:
            param_count += 1
            filters.append(f"event_nature = ANY(${param_count})")
            params.append(event_types)
        
        if countries:
            param_count += 1
            filters.append(f"country = ANY(${param_count})")
            params.append(countries)
        
        if min_confidence > 0:
            param_count += 1
            filters.append(f"extraction_confidence_score >= ${param_count}")
            params.append(min_confidence)
        
        # Add pagination parameters
        param_count += 1
        limit_param = f"${param_count}"
        params.append(limit)
        
        param_count += 1
        offset_param = f"${param_count}"
        params.append(offset)
        
        where_clause = " AND ".join(filters)
        
        query = f"""
            SELECT 
                external_id,
                title,
                content_url,
                published_at,
                event_nature,
                event_summary,
                location_name,
                country,
                latitude,
                longitude,
                fatalities_count,
                casualties_count,
                extraction_confidence_score,
                avg_tone,
                goldstein_scale,
                num_mentions,
                conflict_classification,
                created_at
            FROM information_units
            WHERE unit_type LIKE 'gdelt_%' AND {where_clause}
            ORDER BY published_at DESC, extraction_confidence_score DESC NULLS LAST
            LIMIT {limit_param} OFFSET {offset_param}
        """
        
        # Count query for total
        count_query = f"""
            SELECT COUNT(*)
            FROM information_units
            WHERE unit_type LIKE 'gdelt_%' AND {where_clause}
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            # Execute main query
            rows = await conn.fetch(query, *params)
            
            # Execute count query (remove limit/offset params)
            total_count = await conn.fetchval(count_query, *params[:-2])
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=total_count,
            execution_time_ms=execution_time,
            query_metadata={
                'filters_applied': len(filters),
                'date_range_days': (end_date - start_date).days,
                'pagination': {'limit': limit, 'offset': offset}
            }
        )
    
    async def get_event_by_id(self, external_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific event by external ID."""
        query = """
            SELECT *
            FROM information_units
            WHERE external_id = $1 AND unit_type LIKE 'gdelt_%'
        """
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, external_id)
            return dict(row) if row else None
    
    # ==================== CONFLICT QUERIES ====================
    
    async def get_conflict_events(
        self,
        days_back: int = 7,
        severity_threshold: float = -5.0,
        min_fatalities: int = 0,
        countries: Optional[List[str]] = None,
        limit: int = 500
    ) -> QueryResult:
        """
        Get conflict-related events with severity filtering.
        
        Args:
            days_back: Number of days to look back
            severity_threshold: Goldstein scale threshold (negative = more severe)
            min_fatalities: Minimum fatality count
            countries: List of countries to filter by
            limit: Maximum number of results
            
        Returns:
            QueryResult with conflict events
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        filters = [
            "published_at >= $1",
            "unit_type LIKE 'gdelt_%'",
            "(conflict_classification IS NOT NULL OR goldstein_scale <= $2 OR fatalities_count >= $3)"
        ]
        params = [start_date, severity_threshold, min_fatalities]
        param_count = 3
        
        if countries:
            param_count += 1
            filters.append(f"country = ANY(${param_count})")
            params.append(countries)
        
        param_count += 1
        params.append(limit)
        
        where_clause = " AND ".join(filters)
        
        query = f"""
            SELECT 
                external_id,
                title,
                content_url,
                published_at,
                event_nature,
                event_summary,
                location_name,
                country,
                latitude,
                longitude,
                fatalities_count,
                casualties_count,
                people_displaced,
                conflict_classification,
                event_severity,
                goldstein_scale,
                avg_tone,
                extraction_confidence_score,
                weapons_methods,
                primary_actors,
                secondary_actors,
                created_at
            FROM information_units
            WHERE {where_clause}
            ORDER BY 
                COALESCE(fatalities_count, 0) DESC,
                COALESCE(goldstein_scale, 0) ASC,
                published_at DESC
            LIMIT ${param_count}
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={
                'days_back': days_back,
                'severity_threshold': severity_threshold,
                'min_fatalities': min_fatalities
            }
        )
    
    async def get_conflict_hotspots(
        self,
        days_back: int = 30,
        min_events: int = 5,
        radius_km: float = 50.0
    ) -> QueryResult:
        """
        Identify conflict hotspots by geographic clustering.
        
        Args:
            days_back: Number of days to analyze
            min_events: Minimum events to qualify as hotspot
            radius_km: Radius in kilometers for clustering
            
        Returns:
            QueryResult with hotspot locations
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        query = """
            WITH conflict_events AS (
                SELECT 
                    external_id,
                    location_name,
                    country,
                    latitude,
                    longitude,
                    fatalities_count,
                    conflict_classification,
                    published_at
                FROM information_units
                WHERE published_at >= $1
                    AND unit_type LIKE 'gdelt_%'
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                    AND (conflict_classification IS NOT NULL 
                         OR goldstein_scale <= -3.0 
                         OR fatalities_count > 0)
            ),
            hotspot_clusters AS (
                SELECT 
                    country,
                    location_name,
                    AVG(latitude) as center_lat,
                    AVG(longitude) as center_lng,
                    COUNT(*) as event_count,
                    SUM(COALESCE(fatalities_count, 0)) as total_fatalities,
                    MIN(published_at) as first_event,
                    MAX(published_at) as latest_event,
                    array_agg(DISTINCT conflict_classification) FILTER (WHERE conflict_classification IS NOT NULL) as conflict_types
                FROM conflict_events
                GROUP BY country, location_name
                HAVING COUNT(*) >= $2
            )
            SELECT 
                country,
                location_name,
                center_lat,
                center_lng,
                event_count,
                total_fatalities,
                first_event,
                latest_event,
                conflict_types,
                ROUND((event_count::float / $3) * 100, 2) as events_per_day
            FROM hotspot_clusters
            ORDER BY event_count DESC, total_fatalities DESC
            LIMIT 100
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, start_date, min_events, days_back)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={
                'analysis_period_days': days_back,
                'min_events_threshold': min_events,
                'clustering_radius_km': radius_km
            }
        )
    
    # ==================== GEOGRAPHIC QUERIES ====================
    
    async def get_events_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0,
        days_back: int = 30,
        limit: int = 100
    ) -> QueryResult:
        """
        Get events within a geographic radius.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Search radius in kilometers
            days_back: Number of days to search back
            limit: Maximum number of results
            
        Returns:
            QueryResult with nearby events
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        # Haversine distance calculation in PostgreSQL
        query = """
            SELECT 
                external_id,
                title,
                content_url,
                published_at,
                event_nature,
                event_summary,
                location_name,
                country,
                latitude,
                longitude,
                fatalities_count,
                casualties_count,
                conflict_classification,
                extraction_confidence_score,
                -- Calculate distance in kilometers
                (6371 * acos(
                    cos(radians($1)) * cos(radians(latitude)) * 
                    cos(radians(longitude) - radians($2)) + 
                    sin(radians($1)) * sin(radians(latitude))
                )) AS distance_km
            FROM information_units
            WHERE published_at >= $3
                AND unit_type LIKE 'gdelt_%'
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
                AND latitude BETWEEN $1 - ($4 / 111.0) AND $1 + ($4 / 111.0)
                AND longitude BETWEEN $2 - ($4 / (111.0 * cos(radians($1)))) AND $2 + ($4 / (111.0 * cos(radians($1))))
            HAVING (6371 * acos(
                cos(radians($1)) * cos(radians(latitude)) * 
                cos(radians(longitude) - radians($2)) + 
                sin(radians($1)) * sin(radians(latitude))
            )) <= $4
            ORDER BY distance_km ASC, published_at DESC
            LIMIT $5
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, latitude, longitude, start_date, radius_km, limit)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={
                'center_coordinates': [latitude, longitude],
                'search_radius_km': radius_km,
                'search_period_days': days_back
            }
        )
    
    async def get_country_statistics(
        self,
        countries: Optional[List[str]] = None,
        days_back: int = 30
    ) -> QueryResult:
        """
        Get aggregated statistics by country.
        
        Args:
            countries: List of countries to analyze (None for all)
            days_back: Number of days to analyze
            
        Returns:
            QueryResult with country statistics
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        country_filter = ""
        params = [start_date]
        
        if countries:
            country_filter = "AND country = ANY($2)"
            params.append(countries)
        
        query = f"""
            SELECT 
                country,
                COUNT(*) as total_events,
                COUNT(*) FILTER (WHERE conflict_classification IS NOT NULL) as conflict_events,
                SUM(COALESCE(fatalities_count, 0)) as total_fatalities,
                SUM(COALESCE(casualties_count, 0)) as total_casualties,
                SUM(COALESCE(people_displaced, 0)) as total_displaced,
                AVG(COALESCE(goldstein_scale, 0)) as avg_goldstein_scale,
                AVG(COALESCE(avg_tone, 0)) as avg_tone,
                AVG(COALESCE(extraction_confidence_score, 0)) as avg_confidence,
                COUNT(DISTINCT location_name) as unique_locations,
                MIN(published_at) as earliest_event,
                MAX(published_at) as latest_event
            FROM information_units
            WHERE published_at >= $1
                AND unit_type LIKE 'gdelt_%'
                AND country IS NOT NULL
                {country_filter}
            GROUP BY country
            ORDER BY total_events DESC, total_fatalities DESC
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={
                'analysis_period_days': days_back,
                'countries_filter': countries
            }
        )
    
    # ==================== TEMPORAL QUERIES ====================
    
    async def get_event_timeline(
        self,
        event_types: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        days_back: int = 30,
        grouping: str = 'daily'  # 'hourly', 'daily', 'weekly'
    ) -> QueryResult:
        """
        Get event timeline with temporal aggregation.
        
        Args:
            event_types: List of event types to include
            countries: List of countries to include
            days_back: Number of days to analyze
            grouping: Temporal grouping ('hourly', 'daily', 'weekly')
            
        Returns:
            QueryResult with timeline data
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        # Determine date truncation based on grouping
        date_trunc_format = {
            'hourly': 'hour',
            'daily': 'day',
            'weekly': 'week'
        }.get(grouping, 'day')
        
        filters = ["published_at >= $1", "unit_type LIKE 'gdelt_%'"]
        params = [start_date]
        param_count = 1
        
        if event_types:
            param_count += 1
            filters.append(f"event_nature = ANY(${param_count})")
            params.append(event_types)
        
        if countries:
            param_count += 1
            filters.append(f"country = ANY(${param_count})")
            params.append(countries)
        
        where_clause = " AND ".join(filters)
        
        query = f"""
            SELECT 
                DATE_TRUNC('{date_trunc_format}', published_at) as time_period,
                COUNT(*) as event_count,
                COUNT(*) FILTER (WHERE conflict_classification IS NOT NULL) as conflict_count,
                SUM(COALESCE(fatalities_count, 0)) as total_fatalities,
                SUM(COALESCE(casualties_count, 0)) as total_casualties,
                AVG(COALESCE(goldstein_scale, 0)) as avg_goldstein_scale,
                AVG(COALESCE(avg_tone, 0)) as avg_tone,
                COUNT(DISTINCT country) as countries_affected,
                COUNT(DISTINCT location_name) as unique_locations
            FROM information_units
            WHERE {where_clause}
            GROUP BY DATE_TRUNC('{date_trunc_format}', published_at)
            ORDER BY time_period DESC
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={
                'grouping': grouping,
                'analysis_period_days': days_back,
                'event_types_filter': event_types,
                'countries_filter': countries
            }
        )
    
    # ==================== ML QUERIES ====================
    
    async def get_high_confidence_events(
        self,
        min_confidence: float = 0.8,
        days_back: int = 7,
        limit: int = 100
    ) -> QueryResult:
        """
        Get events with high ML confidence scores.
        
        Args:
            min_confidence: Minimum confidence threshold
            days_back: Number of days to search back
            limit: Maximum number of results
            
        Returns:
            QueryResult with high-confidence events
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        query = """
            SELECT 
                external_id,
                title,
                content_url,
                published_at,
                event_nature,
                event_summary,
                location_name,
                country,
                extraction_confidence_score,
                model_version,
                extraction_methodology,
                fatalities_count,
                casualties_count,
                conflict_classification,
                thinking_traces,
                extraction_reasoning
            FROM information_units
            WHERE published_at >= $1
                AND unit_type LIKE 'gdelt_%'
                AND extraction_confidence_score >= $2
                AND extraction_confidence_score IS NOT NULL
            ORDER BY extraction_confidence_score DESC, published_at DESC
            LIMIT $3
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, start_date, min_confidence, limit)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={
                'min_confidence_threshold': min_confidence,
                'search_period_days': days_back
            }
        )
    
    async def get_ml_processing_statistics(self) -> QueryResult:
        """Get ML processing statistics."""
        query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(*) FILTER (WHERE extraction_confidence_score IS NOT NULL) as ml_processed,
                COUNT(*) FILTER (WHERE extraction_confidence_score >= 0.8) as high_confidence,
                COUNT(*) FILTER (WHERE extraction_confidence_score >= 0.6) as medium_confidence,
                COUNT(*) FILTER (WHERE extraction_confidence_score < 0.6) as low_confidence,
                AVG(extraction_confidence_score) as avg_confidence,
                COUNT(DISTINCT model_version) as model_versions_used,
                COUNT(*) FILTER (WHERE thinking_traces IS NOT NULL) as with_thinking_traces,
                COUNT(*) FILTER (WHERE event_nature IS NOT NULL) as with_event_extraction,
                COUNT(*) FILTER (WHERE conflict_classification IS NOT NULL) as with_conflict_analysis
            FROM information_units
            WHERE unit_type LIKE 'gdelt_%'
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row)] if row else [],
            total_count=1,
            execution_time_ms=execution_time,
            query_metadata={'query_type': 'ml_statistics'}
        )
    
    # ==================== UTILITY QUERIES ====================
    
    async def search_events(
        self,
        search_text: str,
        days_back: int = 30,
        limit: int = 100,
        search_fields: List[str] = None
    ) -> QueryResult:
        """
        Full-text search across event fields.
        
        Args:
            search_text: Text to search for
            days_back: Number of days to search back
            limit: Maximum number of results
            search_fields: List of fields to search in
            
        Returns:
            QueryResult with matching events
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        if search_fields is None:
            search_fields = ['title', 'content', 'event_summary', 'location_name']
        
        # Build search conditions
        search_conditions = []
        for field in search_fields:
            search_conditions.append(f"{field} ILIKE $2")
        
        search_clause = " OR ".join(search_conditions)
        search_pattern = f"%{search_text}%"
        
        query = f"""
            SELECT 
                external_id,
                title,
                content_url,
                published_at,
                event_nature,
                event_summary,
                location_name,
                country,
                extraction_confidence_score,
                conflict_classification,
                fatalities_count,
                -- Calculate relevance score based on field matches
                CASE 
                    WHEN title ILIKE $2 THEN 3
                    WHEN event_summary ILIKE $2 THEN 2
                    WHEN content ILIKE $2 THEN 1
                    ELSE 0
                END as relevance_score
            FROM information_units
            WHERE published_at >= $1
                AND unit_type LIKE 'gdelt_%'
                AND ({search_clause})
            ORDER BY relevance_score DESC, extraction_confidence_score DESC NULLS LAST, published_at DESC
            LIMIT $3
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, start_date, search_pattern, limit)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={
                'search_text': search_text,
                'search_fields': search_fields,
                'search_period_days': days_back
            }
        )
    
    async def get_related_events(
        self,
        event_id: str,
        similarity_threshold: float = 0.7,
        limit: int = 10
    ) -> QueryResult:
        """
        Find events related to a specific event.
        
        Args:
            event_id: External ID of the reference event
            similarity_threshold: Minimum similarity threshold
            limit: Maximum number of results
            
        Returns:
            QueryResult with related events
        """
        query = """
            WITH reference_event AS (
                SELECT location_name, country, event_nature, 
                       latitude, longitude, published_at,
                       primary_actors, secondary_actors
                FROM information_units
                WHERE external_id = $1
            )
            SELECT 
                iu.external_id,
                iu.title,
                iu.content_url,
                iu.published_at,
                iu.event_nature,
                iu.event_summary,
                iu.location_name,
                iu.country,
                iu.extraction_confidence_score,
                -- Calculate similarity score
                (
                    CASE WHEN iu.country = re.country THEN 0.3 ELSE 0 END +
                    CASE WHEN iu.location_name = re.location_name THEN 0.3 ELSE 0 END +
                    CASE WHEN iu.event_nature = re.event_nature THEN 0.2 ELSE 0 END +
                    CASE WHEN ABS(EXTRACT(EPOCH FROM (iu.published_at - re.published_at))) < 86400 THEN 0.2 ELSE 0 END
                ) as similarity_score
            FROM information_units iu, reference_event re
            WHERE iu.external_id != $1
                AND iu.unit_type LIKE 'gdelt_%'
                AND iu.published_at BETWEEN re.published_at - INTERVAL '7 days' 
                                       AND re.published_at + INTERVAL '7 days'
            HAVING (
                CASE WHEN iu.country = re.country THEN 0.3 ELSE 0 END +
                CASE WHEN iu.location_name = re.location_name THEN 0.3 ELSE 0 END +
                CASE WHEN iu.event_nature = re.event_nature THEN 0.2 ELSE 0 END +
                CASE WHEN ABS(EXTRACT(EPOCH FROM (iu.published_at - re.published_at))) < 86400 THEN 0.2 ELSE 0 END
            ) >= $2
            ORDER BY similarity_score DESC, iu.published_at DESC
            LIMIT $3
        """
        
        start_time = datetime.now()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, event_id, similarity_threshold, limit)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={
                'reference_event_id': event_id,
                'similarity_threshold': similarity_threshold
            }
        )


# Utility functions
async def execute_custom_query(
    pool: asyncpg.Pool,
    query: str,
    params: List[Any] = None
) -> QueryResult:
    """
    Execute a custom query with timing and error handling.
    
    Args:
        pool: Database connection pool
        query: SQL query to execute
        params: Query parameters
        
    Returns:
        QueryResult with results and metadata
    """
    start_time = datetime.now()
    params = params or []
    
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResult(
            data=[dict(row) for row in rows],
            total_count=len(rows),
            execution_time_ms=execution_time,
            query_metadata={'custom_query': True}
        )
    
    except Exception as e:
        logger.error(f"Custom query failed: {e}")
        raise


# Export main classes
__all__ = [
    'QueryResult',
    'GDELTQueryOptimizer',
    'execute_custom_query'
]