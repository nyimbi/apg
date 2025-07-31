#!/usr/bin/env python3
"""
Import Last 2 Weeks of GDELT Data
=================================

This script uses the BigQuery ETL integration to import the last 2 weeks of
GDELT GKG data into the information_units table, focusing on Horn of Africa.

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

async def import_last_2_weeks():
    """Import the last 2 weeks of GDELT data using BigQuery ETL."""
    
    try:
        # Set up authentication
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/nyimbiodero/.config/gcloud/application_default_credentials.json'
        
        # Import BigQuery ETL directly
        from database.bigquery_etl import GDELTBigQueryETL, BigQueryETLConfig
        
        print("🚀 IMPORTING LAST 2 WEEKS OF GDELT DATA")
        print("=" * 60)
        
        # Calculate date range (last 2 weeks)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        print(f"📅 Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("🌍 Target: Horn of Africa countries (ET, SO, ER, DJ, SS, SD, KE, UG)")
        print("📊 Mode: BigQuery direct access")
        print("🔧 Batch Size: 3,000 records per batch")
        print()
        
        # Create BigQuery ETL configuration
        config = BigQueryETLConfig(
            database_url="postgresql:///lnd",
            bigquery_project="gdelt-bq",
            google_credentials_path="/Users/nyimbiodero/.config/gcloud/application_default_credentials.json",
            target_countries=["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"],
            batch_size=3000,  # Larger batches for efficiency
            max_concurrent_batches=2,
            enable_ml_processing=False,  # Disable ML for faster processing
            enable_geographic_filtering=True,
            skip_existing_records=True,  # Skip duplicates
            use_bulk_operations=True
        )
        
        print("✅ Configuration created")
        
        # Initialize ETL
        etl = GDELTBigQueryETL(config)
        await etl.initialize()
        print("✅ BigQuery ETL initialized")
        
        # Perform health check
        healthy = await etl.health_check()
        if not healthy:
            print("❌ Health check failed - aborting import")
            return False
        print("✅ Health check passed")
        
        # Progress callback function
        def progress_callback(metrics):
            print(f"📈 Progress: {metrics.total_records_processed:,} records processed "
                  f"({metrics.records_per_second:.1f} records/sec)")
        
        print(f"\n🔄 Starting import for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print("This may take several minutes...")
        
        # Process the data
        metrics = await etl.process_date_range(
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback
        )
        
        print(f"\n🎉 IMPORT COMPLETED!")
        print("=" * 60)
        print(f"📊 FINAL RESULTS:")
        print(f"  Total Records Processed: {metrics.total_records_processed:,}")
        print(f"  Records Inserted: {metrics.records_inserted:,}")
        print(f"  Records Updated: {metrics.records_updated:,}")
        print(f"  Records Skipped: {metrics.records_skipped:,}")
        print(f"  Records Failed: {metrics.records_failed:,}")
        print(f"  Processing Time: {metrics.duration_seconds:.1f} seconds")
        print(f"  Processing Rate: {metrics.records_per_second:.1f} records/second")
        print(f"  BigQuery Queries: {metrics.bigquery_queries_executed}")
        print(f"  BigQuery Bytes Processed: {metrics.bigquery_bytes_processed:,}")
        
        # Verify the import in database
        import asyncpg
        conn = await asyncpg.connect("postgresql:///lnd")
        
        # Check total records in information_units
        total_count = await conn.fetchval("SELECT COUNT(*) FROM information_units")
        
        # Check records from last 2 weeks
        recent_count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_units 
            WHERE created_at >= $1
        """, start_date)
        
        # Check by country
        country_stats = await conn.fetch("""
            SELECT country, COUNT(*) as count
            FROM information_units 
            WHERE created_at >= $1 AND country IS NOT NULL
            GROUP BY country
            ORDER BY count DESC
            LIMIT 10
        """, start_date)
        
        # Check conflict-related records
        conflict_count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_units 
            WHERE created_at >= $1 AND is_conflict_related = true
        """, start_date)
        
        await conn.close()
        
        print(f"\n📊 DATABASE VERIFICATION:")
        print(f"  Total information_units: {total_count:,}")
        print(f"  New records (last 2 weeks): {recent_count:,}")
        print(f"  Conflict-related records: {conflict_count:,}")
        
        if country_stats:
            print(f"\n🌍 TOP COUNTRIES (last 2 weeks):")
            for row in country_stats:
                print(f"  {row['country']}: {row['count']:,} records")
        
        await etl.close()
        print("\n✅ ETL pipeline closed")
        
        # Calculate success rate
        if metrics.total_records_processed > 0:
            success_rate = ((metrics.records_inserted + metrics.records_updated) / 
                          metrics.total_records_processed) * 100
            print(f"\n📈 Success Rate: {success_rate:.1f}%")
            
            if success_rate > 90:
                print("🏆 Excellent import performance!")
            elif success_rate > 75:
                print("✅ Good import performance")
            else:
                print("⚠️  Import completed with some issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(import_last_2_weeks())
    if success:
        print(f"\n🎉 SUCCESS: Last 2 weeks of GDELT data imported successfully!")
        print("The information_units table has been updated with the latest global events data.")
    else:
        print(f"\n❌ FAILED: Import was not successful")