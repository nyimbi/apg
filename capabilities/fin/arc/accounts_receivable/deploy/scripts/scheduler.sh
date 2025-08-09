#!/bin/bash
# APG Accounts Receivable - Scheduler Process Startup Script
# © 2025 Datacraft. All rights reserved.

set -euo pipefail

# Configuration
export APG_AR_LOG_LEVEL=${APG_AR_LOG_LEVEL:-INFO}
export SCHEDULER_CHECK_INTERVAL=${SCHEDULER_CHECK_INTERVAL:-60}

# Logging setup
exec > >(tee -a /var/log/ar/scheduler.log)
exec 2>&1

echo "⏰ Starting APG Accounts Receivable Scheduler"
echo "============================================="
echo "Environment: ${APG_ENVIRONMENT:-production}"
echo "Log Level: ${APG_AR_LOG_LEVEL}"
echo "Check Interval: ${SCHEDULER_CHECK_INTERVAL}s"
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
echo "🔧 Validating scheduler configuration..."
cd /opt/apg/ar
python -c "
import sys
sys.path.insert(0, '/opt/apg')
from ar.service import ARService
try:
    service = ARService()
    print('✅ Scheduler configuration is valid')
except Exception as e:
    print(f'❌ Scheduler configuration validation failed: {e}')
    sys.exit(1)
" || exit 1

# Create required directories
mkdir -p /var/log/ar /opt/apg/data/scheduler

# Start the scheduler
echo "📅 Starting APG AR scheduled tasks..."

cd /opt/apg

# Scheduler implementation
python -c "
import asyncio
import signal
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/apg')
from ar.service import ARService

class ARScheduler:
    def __init__(self):
        self.running = True
        self.service = ARService()
        self.last_runs = {}
        
    async def run_scheduler(self):
        \"\"\"Main scheduler loop.\"\"\"
        print(f'📅 Scheduler started at {datetime.now()}')
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Run scheduled tasks based on frequency
                await self.check_hourly_tasks(current_time)
                await self.check_daily_tasks(current_time)
                await self.check_weekly_tasks(current_time)
                await self.check_monthly_tasks(current_time)
                
                # Sleep until next check
                await asyncio.sleep(${SCHEDULER_CHECK_INTERVAL})
                
            except Exception as e:
                print(f'❌ Scheduler error: {e}')
                await asyncio.sleep(60)  # Wait before retrying
                
        print('🛑 Scheduler stopped')
    
    async def check_hourly_tasks(self, current_time):
        \"\"\"Check and run hourly scheduled tasks.\"\"\"
        if self.should_run('hourly', current_time, timedelta(hours=1)):
            print(f'⏰ Running hourly tasks at {current_time}')
            
            try:
                # Automatic overdue invoice marking
                await self.mark_overdue_invoices()
                
                # Send payment reminders
                await self.send_payment_reminders()
                
                # Process payment matching queue
                await self.process_payment_matching()
                
                self.last_runs['hourly'] = current_time
                
            except Exception as e:
                print(f'❌ Hourly tasks error: {e}')
    
    async def check_daily_tasks(self, current_time):
        \"\"\"Check and run daily scheduled tasks.\"\"\"
        if self.should_run('daily', current_time, timedelta(days=1)):
            print(f'📅 Running daily tasks at {current_time}')
            
            try:
                # Generate daily AR reports
                await self.generate_daily_reports()
                
                # Process collections optimization
                await self.optimize_collections_strategies()
                
                # Update credit assessments
                await self.refresh_credit_assessments()
                
                # Clean up old logs and temp files
                await self.cleanup_old_files()
                
                self.last_runs['daily'] = current_time
                
            except Exception as e:
                print(f'❌ Daily tasks error: {e}')
    
    async def check_weekly_tasks(self, current_time):
        \"\"\"Check and run weekly scheduled tasks.\"\"\"
        if self.should_run('weekly', current_time, timedelta(weeks=1)):
            print(f'📊 Running weekly tasks at {current_time}')
            
            try:
                # Generate weekly AR analytics
                await self.generate_weekly_analytics()
                
                # Process cash flow forecasting
                await self.update_cash_flow_forecasts()
                
                # Database maintenance
                await self.database_maintenance()
                
                self.last_runs['weekly'] = current_time
                
            except Exception as e:
                print(f'❌ Weekly tasks error: {e}')
    
    async def check_monthly_tasks(self, current_time):
        \"\"\"Check and run monthly scheduled tasks.\"\"\"
        if self.should_run('monthly', current_time, timedelta(days=30)):
            print(f'📈 Running monthly tasks at {current_time}')
            
            try:
                # Generate monthly AR reports
                await self.generate_monthly_reports()
                
                # Archive old data
                await self.archive_old_data()
                
                # Performance optimization
                await self.optimize_database_performance()
                
                self.last_runs['monthly'] = current_time
                
            except Exception as e:
                print(f'❌ Monthly tasks error: {e}')
    
    def should_run(self, task_type, current_time, interval):
        \"\"\"Check if a task should run based on its schedule.\"\"\"
        last_run = self.last_runs.get(task_type)
        if last_run is None:
            return True
        return current_time - last_run >= interval
    
    async def mark_overdue_invoices(self):
        \"\"\"Mark invoices as overdue based on due dates.\"\"\"
        try:
            # This would query for invoices past due date
            # and update their status to OVERDUE
            print('📋 Marking overdue invoices...')
        except Exception as e:
            print(f'❌ Mark overdue error: {e}')
    
    async def send_payment_reminders(self):
        \"\"\"Send automated payment reminders.\"\"\"
        try:
            # This would send payment reminders for overdue invoices
            # based on collections rules
            print('📧 Sending payment reminders...')
        except Exception as e:
            print(f'❌ Payment reminders error: {e}')
    
    async def process_payment_matching(self):
        \"\"\"Process automatic payment matching.\"\"\"
        try:
            # This would attempt to match unmatched payments
            # to outstanding invoices
            print('🔄 Processing payment matching...')
        except Exception as e:
            print(f'❌ Payment matching error: {e}')
    
    async def generate_daily_reports(self):
        \"\"\"Generate daily AR reports.\"\"\"
        try:
            # This would generate and email daily AR reports
            print('📊 Generating daily reports...')
        except Exception as e:
            print(f'❌ Daily reports error: {e}')
    
    async def optimize_collections_strategies(self):
        \"\"\"Optimize collections strategies using AI.\"\"\"
        try:
            # This would run AI collections optimization
            # for active collections cases
            print('🤖 Optimizing collections strategies...')
        except Exception as e:
            print(f'❌ Collections optimization error: {e}')
    
    async def refresh_credit_assessments(self):
        \"\"\"Refresh customer credit assessments.\"\"\"
        try:
            # This would refresh credit scores for customers
            # who haven't been assessed recently
            print('💳 Refreshing credit assessments...')
        except Exception as e:
            print(f'❌ Credit assessments error: {e}')
    
    async def cleanup_old_files(self):
        \"\"\"Clean up old log files and temporary data.\"\"\"
        try:
            # This would clean up old logs, temp files, etc.
            print('🧹 Cleaning up old files...')
        except Exception as e:
            print(f'❌ Cleanup error: {e}')
    
    async def generate_weekly_analytics(self):
        \"\"\"Generate weekly analytics reports.\"\"\"
        try:
            print('📈 Generating weekly analytics...')
        except Exception as e:
            print(f'❌ Weekly analytics error: {e}')
    
    async def update_cash_flow_forecasts(self):
        \"\"\"Update cash flow forecasting models.\"\"\"
        try:
            print('💰 Updating cash flow forecasts...')
        except Exception as e:
            print(f'❌ Cash flow forecasts error: {e}')
    
    async def database_maintenance(self):
        \"\"\"Perform database maintenance tasks.\"\"\"
        try:
            print('🗄️ Performing database maintenance...')
        except Exception as e:
            print(f'❌ Database maintenance error: {e}')
    
    async def generate_monthly_reports(self):
        \"\"\"Generate monthly reports.\"\"\"
        try:
            print('📅 Generating monthly reports...')
        except Exception as e:
            print(f'❌ Monthly reports error: {e}')
    
    async def archive_old_data(self):
        \"\"\"Archive old data for compliance.\"\"\"
        try:
            print('📦 Archiving old data...')
        except Exception as e:
            print(f'❌ Data archiving error: {e}')
    
    async def optimize_database_performance(self):
        \"\"\"Optimize database performance.\"\"\"
        try:
            print('⚡ Optimizing database performance...')
        except Exception as e:
            print(f'❌ Database optimization error: {e}')
    
    def stop(self):
        \"\"\"Gracefully stop the scheduler.\"\"\"
        print('🛑 Stopping scheduler...')
        self.running = False

# Signal handlers for graceful shutdown
scheduler = ARScheduler()

def signal_handler(signum, frame):
    print(f'📡 Received signal {signum}')
    scheduler.stop()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Run the scheduler
try:
    asyncio.run(scheduler.run_scheduler())
except KeyboardInterrupt:
    print('👋 Scheduler interrupted')
except Exception as e:
    print(f'💥 Scheduler crashed: {e}')
    sys.exit(1)
"