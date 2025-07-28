#!/bin/bash

# APG Time & Attendance Capability Development Setup Script
# Copyright © 2025 Datacraft

set -e

echo "🚀 Setting up APG Time & Attendance Development Environment"
echo "============================================================"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Create directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p logs data redis grafana/provisioning prometheus

# Create Redis configuration
echo "⚙️ Creating Redis configuration..."
cat > redis/redis.conf << EOF
# Redis configuration for Time & Attendance
port 6379
bind 0.0.0.0
protected-mode no
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
maxmemory 256mb
maxmemory-policy allkeys-lru
EOF

# Create Prometheus configuration
echo "📊 Creating Prometheus configuration..."
cat > prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'time-attendance-app'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF

# Pull Docker images
echo "📥 Pulling Docker images..."
docker-compose pull

# Start services
echo "🚀 Starting services..."
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker-compose exec postgres pg_isready -U ta_user -d time_attendance_db >/dev/null 2>&1; do
    echo "   PostgreSQL is not ready yet, waiting 5 seconds..."
    sleep 5
done

echo "✅ PostgreSQL is ready!"

# Run database migrations
echo "🔄 Running database migrations..."
# Note: This will be run when the app container starts

# Start all services
echo "🌟 Starting all services..."
docker-compose up -d

# Wait for application to be ready
echo "⏳ Waiting for application to be ready..."
timeout=60
count=0
while [ $count -lt $timeout ]; do
    if curl -f http://localhost:8000/api/human_capital_management/time_attendance/health >/dev/null 2>&1; then
        echo "✅ Application is ready!"
        break
    fi
    echo "   Application is not ready yet, waiting 5 seconds..."
    sleep 5
    count=$((count + 5))
done

if [ $count -ge $timeout ]; then
    echo "❌ Application failed to start within $timeout seconds"
    echo "   Check logs with: docker-compose logs app"
    exit 1
fi

echo ""
echo "🎉 Development environment setup complete!"
echo "=========================================="
echo ""
echo "📍 Service URLs:"
echo "   • Application API: http://localhost:8000"
echo "   • API Documentation: http://localhost:8000/api/human_capital_management/time_attendance/docs"
echo "   • pgAdmin: http://localhost:8080 (admin@datacraft.co.ke / admin_secure_2025)"
echo "   • Grafana: http://localhost:3000 (admin / grafana_admin_2025)"
echo "   • Prometheus: http://localhost:9090"
echo ""
echo "📋 Database Info:"
echo "   • Host: localhost:5432"
echo "   • Database: time_attendance_db"
echo "   • Username: ta_user"
echo "   • Password: ta_secure_password_2025"
echo ""
echo "🛠️ Useful Commands:"
echo "   • View logs: docker-compose logs -f [service_name]"
echo "   • Stop services: docker-compose down"
echo "   • Restart services: docker-compose restart [service_name]"
echo "   • Access database: docker-compose exec postgres psql -U ta_user -d time_attendance_db"
echo ""
echo "🔍 To monitor the setup:"
echo "   docker-compose ps"
echo ""