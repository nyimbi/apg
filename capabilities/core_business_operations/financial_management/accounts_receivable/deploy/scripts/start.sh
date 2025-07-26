#!/bin/bash
# APG Accounts Receivable - Production Startup Script
# © 2025 Datacraft. All rights reserved.

set -euo pipefail

# Configuration
export APG_AR_LOG_LEVEL=${APG_AR_LOG_LEVEL:-INFO}
export APG_AR_WORKERS=${APG_AR_WORKERS:-4}
export APG_AR_HOST=${APG_AR_HOST:-0.0.0.0}
export APG_AR_PORT=${APG_AR_PORT:-8000}
export APG_AR_RELOAD=${APG_AR_RELOAD:-false}

# Logging setup
exec > >(tee -a /var/log/ar/startup.log)
exec 2>&1

echo "🚀 Starting APG Accounts Receivable API Server"
echo "=============================================="
echo "Environment: ${APG_ENVIRONMENT:-production}"
echo "Log Level: ${APG_AR_LOG_LEVEL}"
echo "Workers: ${APG_AR_WORKERS}"
echo "Host: ${APG_AR_HOST}"
echo "Port: ${APG_AR_PORT}"
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

# Database migrations
echo "🗄️ Running database migrations..."
cd /opt/apg/ar
python -m alembic upgrade head || {
	echo "❌ Database migration failed"
	exit 1
}

# Validate configuration
echo "🔧 Validating configuration..."
python -c "
import sys
sys.path.insert(0, '/opt/apg')
from ar.service import ARService
try:
    service = ARService()
    print('✅ Configuration is valid')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
" || exit 1

# Create required directories
mkdir -p /var/log/ar /opt/apg/data/uploads /opt/apg/data/exports

# Set file permissions
chmod 755 /var/log/ar /opt/apg/data
chmod 644 /var/log/ar/*.log 2>/dev/null || true

# Start the application
echo "🌟 Starting APG AR API server..."
echo "Command: uvicorn ar.api_endpoints:app --host $APG_AR_HOST --port $APG_AR_PORT --workers $APG_AR_WORKERS --log-level ${APG_AR_LOG_LEVEL,,}"
echo ""

cd /opt/apg

# Production startup with uvicorn
exec uvicorn ar.api_endpoints:app \
	--host "$APG_AR_HOST" \
	--port "$APG_AR_PORT" \
	--workers "$APG_AR_WORKERS" \
	--log-level "${APG_AR_LOG_LEVEL,,}" \
	--access-log \
	--use-colors \
	--server-header \
	--date-header \
	--forwarded-allow-ips='*' \
	--proxy-headers