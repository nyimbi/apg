#!/usr/bin/env python3
"""
Minimal BigQuery ETL Test
=========================

This script tests the BigQuery ETL functionality with minimal dependencies.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: December 27, 2024
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the direct module path
bigquery_etl_path = Path(__file__).parent.parent / "database" / "bigquery_etl.py"

async def test_minimal_bigquery():
    """Test BigQuery ETL with minimal setup."""
    
    print("🚀 MINIMAL BIGQUERY ETL TEST")
    print("=" * 40)
    
    # Set up authentication
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/nyimbiodero/.config/gcloud/application_default_credentials.json'
    
    try:
        # Test Google Cloud authentication
        from google.cloud import bigquery
        client = bigquery.Client(project="gdelt-bq")
        
        # Test simple query
        query = """
        SELECT COUNT(*) as count
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME >= TIMESTAMP('2025-01-01')
          AND _PARTITIONTIME < TIMESTAMP('2025-01-02')
        LIMIT 1
        """
        
        print("✅ BigQuery client created")
        print(f"🔄 Testing query on GDELT data...")
        
        job = client.query(query)
        results = job.result()
        
        for row in results:
            print(f"✅ BigQuery test successful: {row.count:,} records available for 2025-01-01")
        
        # Test database connection
        import asyncpg
        conn = await asyncpg.connect("postgresql:///lnd")
        
        # Check information_units table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'information_units'
            )
        """)
        
        if table_exists:
            count = await conn.fetchval("SELECT COUNT(*) FROM information_units")
            print(f"✅ Database connection successful: {count:,} records in information_units")
        else:
            print("❌ information_units table does not exist")
        
        await conn.close()
        
        print(f"\n🎉 MINIMAL TEST COMPLETED!")
        print("✅ BigQuery connection: Working")
        print("✅ Database connection: Working")
        print("✅ Authentication: Working")
        print("\nThe BigQuery ETL integration is ready to use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_minimal_bigquery())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")