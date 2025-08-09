#!/bin/bash
# APG Accounts Receivable - Worker Process Startup Script
# © 2025 Datacraft. All rights reserved.

set -euo pipefail

# Configuration
export APG_AR_LOG_LEVEL=${APG_AR_LOG_LEVEL:-INFO}
export WORKER_CONCURRENCY=${WORKER_CONCURRENCY:-4}
export WORKER_QUEUE=${WORKER_QUEUE:-default}
export WORKER_MAX_TASKS_PER_CHILD=${WORKER_MAX_TASKS_PER_CHILD:-1000}

# Logging setup
exec > >(tee -a /var/log/ar/worker.log)
exec 2>&1

echo "👷 Starting APG Accounts Receivable Worker"
echo "=========================================="
echo "Environment: ${APG_ENVIRONMENT:-production}"
echo "Log Level: ${APG_AR_LOG_LEVEL}"
echo "Concurrency: ${WORKER_CONCURRENCY}"
echo "Queue: ${WORKER_QUEUE}"
echo "Max Tasks per Child: ${WORKER_MAX_TASKS_PER_CHILD}"
echo "Timestamp: $(date -Iseconds)"
echo ""

# Health check function
health_check() {
	local service=$1
	local host=$2
	local port=$3
	local timeout=${4:-30}
	
	echo "⏳ Waiting for $service to be ready..."
	
	for i in $(seq 1 $timeout); do
		if timeout 1 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
			echo "✅ $service is ready"
			return 0
		fi
		echo "⏳ Waiting for $service... ($i/$timeout)"
		sleep 1
	done
	
	echo "❌ $service is not ready after ${timeout}s"
	return 1
}

# Wait for dependencies
echo "🔍 Checking dependencies..."

# Check database
if [[ -n "${DATABASE_URL:-}" ]]; then
	DB_HOST=$(echo "$DATABASE_URL" | sed -n 's|.*@\([^:]*\):.*|\1|p')
	DB_PORT=$(echo "$DATABASE_URL" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
	if [[ -n "$DB_HOST" && -n "$DB_PORT" ]]; then
		health_check "PostgreSQL" "$DB_HOST" "$DB_PORT" 60
	fi
fi

# Check Redis
if [[ -n "${REDIS_URL:-}" ]]; then
	REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's|redis://\([^:]*\):.*|\1|p')
	REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
	if [[ -n "$REDIS_HOST" && -n "$REDIS_PORT" ]]; then
		health_check "Redis" "$REDIS_HOST" "$REDIS_PORT" 30
	fi
fi

# Validate configuration
echo "🔧 Validating worker configuration..."
cd /opt/apg/ar
python -c "
import sys
sys.path.insert(0, '/opt/apg')
from ar.service import ARService
try:
    service = ARService()
    print('✅ Worker configuration is valid')
except Exception as e:
    print(f'❌ Worker configuration validation failed: {e}')
    sys.exit(1)
" || exit 1

# Create required directories
mkdir -p /var/log/ar /opt/apg/data/worker

# Start the worker processes
echo "⚙️  Starting APG AR background workers..."

cd /opt/apg

# Worker tasks implementation - this would typically use Celery, RQ, or similar
echo "🔄 Starting task processing..."

# Example worker loop (replace with actual task queue implementation)
python -c "
import asyncio
import signal
import sys
import time
from datetime import datetime

sys.path.insert(0, '/opt/apg')
from ar.service import ARService

class ARWorker:
    def __init__(self):
        self.running = True
        self.service = ARService()
        
    async def process_tasks(self):
        \"\"\"Main worker task processing loop.\"\"\"
        print(f'🔄 Worker started at {datetime.now()}')
        
        while self.running:
            try:
                # Process different types of background tasks
                await self.process_credit_assessments()
                await self.process_collections_tasks()
                await self.process_payment_matching()
                await self.process_notifications()
                await self.process_cleanup_tasks()
                
                # Sleep between processing cycles
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f'❌ Worker error: {e}')
                await asyncio.sleep(30)  # Wait before retrying
                
        print('🛑 Worker stopped')
    
    async def process_credit_assessments(self):
        \"\"\"Process queued credit assessments.\"\"\"
        try:
            # This would check for queued credit assessment tasks
            # and process them using the AI credit scoring service
            pass
        except Exception as e:
            print(f'❌ Credit assessment error: {e}')
    
    async def process_collections_tasks(self):
        \"\"\"Process collections automation tasks.\"\"\"
        try:
            # This would process overdue invoice notifications
            # and collections strategy optimization
            pass
        except Exception as e:
            print(f'❌ Collections task error: {e}')
    
    async def process_payment_matching(self):
        \"\"\"Process automatic payment matching.\"\"\"
        try:
            # This would process unmatched payments
            # and attempt automatic invoice application
            pass
        except Exception as e:
            print(f'❌ Payment matching error: {e}')
    
    async def process_notifications(self):
        \"\"\"Process notification queue.\"\"\"
        try:
            # This would send queued notifications
            # (emails, SMS, webhooks, etc.)
            pass
        except Exception as e:
            print(f'❌ Notification error: {e}')
    
    async def process_cleanup_tasks(self):
        \"\"\"Process periodic cleanup tasks.\"\"\"
        try:
            # This would handle log rotation,
            # temporary file cleanup, etc.
            pass
        except Exception as e:
            print(f'❌ Cleanup task error: {e}')
    
    def stop(self):
        \"\"\"Gracefully stop the worker.\"\"\"
        print('🛑 Stopping worker...')
        self.running = False

# Signal handlers for graceful shutdown
worker = ARWorker()

def signal_handler(signum, frame):
    print(f'📡 Received signal {signum}')
    worker.stop()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Run the worker
try:
    asyncio.run(worker.process_tasks())
except KeyboardInterrupt:
    print('👋 Worker interrupted')
except Exception as e:
    print(f'💥 Worker crashed: {e}')
    sys.exit(1)
"