#!/usr/bin/env python3
"""
Import Last 2 Weeks of GDELT Events Data
========================================

This script uses the GDELT Events ETL to import the last 2 weeks of GDELT Events
data into the information_units table. GDELT Events are the core structured event
records that capture who, what, when, where, and how of global events.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: December 27, 2024
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def import_events_last_2_weeks():
    """Import the last 2 weeks of GDELT Events data."""
    
    try:
        # Set up authentication
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/nyimbiodero/.config/gcloud/application_default_credentials.json'
        
        # Import Events ETL directly
        from database.events_etl import GDELTEventsETL, EventsETLConfig
        
        print("🚀 IMPORTING LAST 2 WEEKS OF GDELT EVENTS DATA")
        print("=" * 60)
        
        # Calculate date range (last 2 weeks)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("🌍 Target: Horn of Africa countries (ET, SO, ER, DJ, SS, SD, KE, UG)")
        print("📊 Mode: GDELT Events (structured event data)")
        print("🎯 Destination: information_units table")
        print("🔧 Batch Size: 3,000 events per batch")
        print()
        
        # Create Events ETL configuration
        config = EventsETLConfig(
            database_url="postgresql:///lnd",
            bigquery_project="gdelt-bq",
            google_credentials_path="/Users/nyimbiodero/.config/gcloud/application_default_credentials.json",
            target_countries=["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"],
            batch_size=3000,  # Events are smaller than GKG records
            max_concurrent_batches=2,
            enable_ml_processing=False,  # Disable ML for faster processing
            enable_geographic_filtering=True,
            skip_existing_records=True,  # Skip duplicates
            use_bulk_operations=True
        )
        
        print("✅ Configuration created")
        
        # Initialize ETL
        etl = GDELTEventsETL(config)
        await etl.initialize()
        print("✅ GDELT Events ETL initialized")
        
        # Perform health check
        healthy = await etl.health_check()
        if not healthy:
            print("❌ Health check failed - aborting import")
            return False
        print("✅ Health check passed")
        
        # Progress callback function
        def progress_callback(metrics):
            print(f"📈 Progress: {metrics.total_records_processed:,} events processed "
                  f"({metrics.records_per_second:.1f} events/sec)")
        
        print(f"\n🔄 Starting Events import for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("Processing structured GDELT event data...")
        print("⏱️  Estimated time: 3-7 minutes for 2 weeks of events")
        print()
        
        # Process the data
        metrics = await etl.process_date_range(
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback
        )
        
        print(f"\n🎉 EVENTS IMPORT COMPLETED!")
        print("=" * 60)
        print(f"📊 FINAL RESULTS:")
        print(f"  Total Events Processed: {metrics.total_records_processed:,}")
        print(f"  Events Inserted: {metrics.records_inserted:,}")
        print(f"  Events Updated: {metrics.records_updated:,}")
        print(f"  Events Skipped: {metrics.records_skipped:,}")
        print(f"  Events Failed: {metrics.records_failed:,}")
        print(f"  Processing Time: {metrics.duration_seconds:.1f} seconds")
        print(f"  Processing Rate: {metrics.records_per_second:.1f} events/second")
        print(f"  BigQuery Queries: {metrics.bigquery_queries_executed}")
        print(f"  BigQuery Bytes Processed: {metrics.bigquery_bytes_processed:,}")
        
        # Verify the import in database
        import asyncpg
        conn = await asyncpg.connect("postgresql:///lnd")
        
        # Check total records in information_units
        total_count = await conn.fetchval("SELECT COUNT(*) FROM information_units")
        
        # Check events from last 2 weeks
        events_count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_units 
            WHERE created_at >= $1 AND unit_type = 'gdelt_event'
        """, start_date)
        
        # Check conflict events
        conflict_events = await conn.fetchval("""
            SELECT COUNT(*) FROM information_units 
            WHERE created_at >= $1 AND unit_type = 'gdelt_event' 
            AND conflict_classification = 'conflict'
        """, start_date)
        
        # Check by country
        country_stats = await conn.fetch("""
            SELECT country_extracted as country, COUNT(*) as count
            FROM information_units 
            WHERE created_at >= $1 AND unit_type = 'gdelt_event' AND country_extracted IS NOT NULL
            GROUP BY country_extracted
            ORDER BY count DESC
            LIMIT 10
        """, start_date)
        
        # Check event types
        event_stats = await conn.fetch("""
            SELECT event_nature, COUNT(*) as count
            FROM information_units 
            WHERE created_at >= $1 AND unit_type = 'gdelt_event' AND event_nature IS NOT NULL
            GROUP BY event_nature
            ORDER BY count DESC
            LIMIT 8
        """, start_date)
        
        await conn.close()
        
        print(f"\n📊 DATABASE VERIFICATION:")
        print(f"  Total information_units: {total_count:,}")
        print(f"  New GDELT events (last 2 weeks): {events_count:,}")
        print(f"  Conflict events: {conflict_events:,}")
        
        if country_stats:
            print(f"\n🌍 TOP COUNTRIES (last 2 weeks):")
            for row in country_stats:
                print(f"  {row['country']}: {row['count']:,} events")
        
        if event_stats:
            print(f"\n🎯 TOP EVENT TYPES (last 2 weeks):")
            for row in event_stats:
                event_type = row['event_nature'][:40] + "..." if len(row['event_nature']) > 40 else row['event_nature']
                print(f"  {event_type}: {row['count']:,} events")
        
        await etl.close()
        print("\n✅ ETL pipeline closed")
        
        # Calculate success rate
        if metrics.total_records_processed > 0:
            success_rate = ((metrics.records_inserted + metrics.records_updated) / 
                          metrics.total_records_processed) * 100
            print(f"\n📈 Success Rate: {success_rate:.1f}%")
            
            if success_rate > 90:
                print("🏆 Excellent Events import performance!")
            elif success_rate > 75:
                print("✅ Good Events import performance")
            else:
                print("⚠️  Import completed with some issues")
        
        # Event quality assessment
        if events_count > 0:
            conflict_rate = (conflict_events / events_count) * 100
            print(f"\n🚨 Conflict Analysis:")
            print(f"  Conflict events: {conflict_events:,} ({conflict_rate:.1f}% of total)")
            print(f"  Non-conflict events: {events_count - conflict_events:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🌟 GDELT Events Data Import: Last 2 Weeks to Information Units")
    print("Importing structured event data with actors, locations, and classifications")
    print()
    
    success = asyncio.run(import_events_last_2_weeks())
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print("✅ Last 2 weeks of GDELT Events successfully imported to information_units")
        print("✅ Data includes structured events with actors, locations, and conflict analysis")
        print("✅ Geographic focus on Horn of Africa region")
        print("✅ Ready for advanced intelligence analysis and relationship mapping")
    else:
        print(f"\n❌ FAILED!")
        print("Import was not successful - check logs above for details")