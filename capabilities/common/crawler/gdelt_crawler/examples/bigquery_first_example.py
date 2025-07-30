#!/usr/bin/env python3
"""
GDELT BigQuery-First Example
============================

Demonstrates the updated GDELT crawler with BigQuery as the primary data source
and intelligent fallback capabilities.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

async def main():
    """Demonstrate BigQuery-first GDELT crawler usage."""
    
    try:
        from crawlers.gdelt_crawler import (
            create_gdelt_crawler,
            create_bigquery_gdelt_crawler, 
            run_bigquery_etl,
            query_gdelt_events
        )
        
        print("🚀 GDELT BIGQUERY-FIRST CRAWLER DEMONSTRATION")
        print("=" * 60)
        
        # Database configuration
        database_url = "postgresql:///lnd"  # Adjust as needed
        
        print("\n1️⃣ CREATING GDELT CRAWLER WITH BIGQUERY DEFAULT")
        print("-" * 50)
        
        # Create crawler - BigQuery is now the default!
        crawler = create_gdelt_crawler(
            database_url=database_url,
            target_countries=["ET", "SO", "ER", "DJ"],  # Horn of Africa focus
            use_events_data=True,  # Use structured Events data
            fallback_enabled=True  # Enable intelligent fallback
        )
        
        print("✅ Crawler created with BigQuery as primary method")
        
        # Initialize and check health
        await crawler.start()
        health = await crawler.health_check()
        
        print(f"\n📊 SYSTEM HEALTH CHECK:")
        print(f"Overall Status: {health['overall_status']}")
        for component, status in health['components'].items():
            print(f"  {component}: {status['status']}")
        
        print(f"\n2️⃣ BIGQUERY ETL DEMONSTRATION")
        print("-" * 50)
        
        # Process recent data using BigQuery
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)  # Process last day
        
        print(f"Processing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Run daily ETL (will use BigQuery automatically)
        result = await crawler.run_daily_etl(date=start_date)
        
        print(f"\n📈 ETL RESULTS:")
        print(f"  Success: {result['success']}")
        print(f"  Method Used: {result.get('method_used', 'unknown')}")
        
        if result['success']:
            processed_counts = result.get('processed_counts', {})
            print(f"  Records Processed: {sum(processed_counts.values()):,}")
            for data_type, count in processed_counts.items():
                print(f"    {data_type}: {count:,}")
            
            print(f"  Processing Time: {result.get('duration', 0):.1f} seconds")
        else:
            print(f"  Errors: {result.get('errors', [])}")
        
        await crawler.stop()
        
        print(f"\n3️⃣ BIGQUERY-ONLY CRAWLER DEMONSTRATION")
        print("-" * 50)
        
        # Create BigQuery-only crawler (no fallbacks)
        bigquery_crawler = create_bigquery_gdelt_crawler(
            database_url=database_url,
            target_countries=["ET", "SO"],
            use_events_data=True
        )
        
        await bigquery_crawler.start()
        print("✅ BigQuery-only crawler created and started")
        
        # Process data using BigQuery Events ETL
        events_result = await bigquery_crawler.process_events_data(
            start_date=start_date,
            end_date=end_date
        )
        
        print(f"\n📊 BIGQUERY EVENTS ETL RESULTS:")
        if events_result['success']:
            print(f"  Events Processed: {events_result['total_records_processed']:,}")
            print(f"  Records Inserted: {events_result['records_inserted']:,}")
            print(f"  Records Updated: {events_result['records_updated']:,}")
            print(f"  Processing Rate: {events_result.get('records_per_second', 0):.1f} records/sec")
            print(f"  BigQuery Queries: {events_result.get('bigquery_queries_executed', 0)}")
        else:
            print(f"  ❌ Failed: {events_result.get('error', 'Unknown error')}")
        
        await bigquery_crawler.stop()
        
        print(f"\n4️⃣ CONVENIENCE FUNCTION DEMONSTRATION")
        print("-" * 50)
        
        # Use convenience function for quick BigQuery ETL
        print("Running quick BigQuery ETL...")
        
        quick_result = await run_bigquery_etl(
            database_url=database_url,
            start_date=start_date,
            end_date=end_date,
            target_countries=["ET", "SO", "KE"],
            use_events_data=True
        )
        
        print(f"\n📋 QUICK ETL RESULTS:")
        if quick_result['success']:
            print(f"  Records Processed: {quick_result['total_records_processed']:,}")
            print(f"  Processing Time: {quick_result.get('processing_time_seconds', 0):.1f}s")
        else:
            print(f"  ❌ Failed: {quick_result.get('error', 'Unknown error')}")
        
        print(f"\n5️⃣ CONFLICT QUERY DEMONSTRATION")
        print("-" * 50)
        
        # Query conflict events using BigQuery by default
        print("Querying recent conflict events...")
        
        conflict_events = await query_gdelt_events(
            query="conflict OR violence OR attack",
            days_back=3,
            max_records=10,
            database_url=database_url,
            use_bigquery=True,
            target_countries=["ET", "SO"]
        )
        
        print(f"\n🚨 CONFLICT EVENTS FOUND: {len(conflict_events)}")
        for i, event in enumerate(conflict_events[:3], 1):
            print(f"  {i}. {event.event_description}")
            print(f"     Date: {event.event_date}")
            print(f"     Location: {event.location or 'Unknown'}")
            print(f"     Actors: {event.actor1_name} vs {event.actor2_name}")
            print()
        
        print(f"\n🎉 BIGQUERY-FIRST DEMONSTRATION COMPLETED!")
        print("=" * 60)
        print("🔍 KEY TAKEAWAYS:")
        print("  • BigQuery is now the default data source")
        print("  • Intelligent fallback to API/files if BigQuery fails")
        print("  • GDELT Events provide structured conflict data")
        print("  • Real-time processing with high performance")
        print("  • Seamless integration with information_units table")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")