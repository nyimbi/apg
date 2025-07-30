#!/usr/bin/env python3
"""
Direct BigQuery ETL Test
========================

This script directly tests the BigQuery ETL functionality without going through
the full crawler initialization.

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

async def test_direct_bigquery():
    """Test BigQuery ETL directly."""
    
    try:
        # Set up authentication
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/nyimbiodero/.config/gcloud/application_default_credentials.json'
        
        # Import the BigQuery ETL directly from the file
        from crawlers.gdelt_crawler.database.bigquery_etl import GDELTBigQueryETL, BigQueryETLConfig
        
        print("🚀 TESTING DIRECT BIGQUERY ETL")
        print("=" * 50)
        
        # Create configuration
        config = BigQueryETLConfig(
            database_url="postgresql:///lnd",
            bigquery_project="gdelt-bq",
            google_credentials_path="/Users/nyimbiodero/.config/gcloud/application_default_credentials.json",
            target_countries=["ET", "SO", "ER", "DJ", "SS", "SD", "KE", "UG"],
            batch_size=1000,  # Small batch for testing
            enable_ml_processing=False  # Disable ML for faster testing
        )
        
        # Create ETL instance
        etl = GDELTBigQueryETL(config)
        print("✅ BigQuery ETL instance created")
        
        # Initialize
        await etl.initialize()
        print("✅ ETL initialized")
        
        # Perform health check
        healthy = await etl.health_check()
        print(f"✅ Health check: {'PASSED' if healthy else 'FAILED'}")
        
        # Test with a small date range (1 day)
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 2)
        
        print(f"\n🔄 Processing data for {start_date.strftime('%Y-%m-%d')}")
        
        # Process the data
        metrics = await etl.process_date_range(start_date, end_date)
        
        print(f"\n📈 PROCESSING RESULTS:")
        print(f"  Total Records Processed: {metrics.total_records_processed:,}")
        print(f"  Records Inserted: {metrics.records_inserted:,}")
        print(f"  Records Updated: {metrics.records_updated:,}")
        print(f"  Records Skipped: {metrics.records_skipped:,}")
        print(f"  Records Failed: {metrics.records_failed:,}")
        print(f"  Processing Time: {metrics.duration_seconds:.1f} seconds")
        print(f"  Processing Rate: {metrics.records_per_second:.1f} records/second")
        print(f"  BigQuery Queries: {metrics.bigquery_queries_executed}")
        print(f"  BigQuery Bytes: {metrics.bigquery_bytes_processed:,}")
        
        # Verify in database
        import asyncpg
        conn = await asyncpg.connect("postgresql:///lnd")
        
        # Check information_units table
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_units 
            WHERE data_source_id = 'gdelt_bigquery'
            AND created_at >= $1
        """, start_date)
        
        await conn.close()
        
        print(f"\n📊 DATABASE VERIFICATION:")
        print(f"  Records in information_units: {count:,}")
        
        await etl.close()
        print("✅ ETL closed")
        
        print(f"\n🎉 DIRECT BIGQUERY ETL TEST COMPLETED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_direct_bigquery())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")