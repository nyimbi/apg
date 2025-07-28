# APG Sustainability & ESG Management - Troubleshooting Guide

**Version:** 1.0.0  
**Last Updated:** January 2025  
**Support Level:** Enterprise

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Installation Problems](#installation-problems)
4. [Authentication & Authorization](#authentication--authorization)
5. [Database Issues](#database-issues)
6. [API Problems](#api-problems)
7. [Performance Issues](#performance-issues)
8. [AI/ML Service Issues](#aiml-service-issues)
9. [Real-time Features](#real-time-features)
10. [Dashboard & UI Problems](#dashboard--ui-problems)
11. [Data Import/Export Issues](#data-importexport-issues)
12. [Integration Problems](#integration-problems)
13. [Error Code Reference](#error-code-reference)
14. [Log Analysis](#log-analysis)
15. [Recovery Procedures](#recovery-procedures)
16. [Support Escalation](#support-escalation)

---

## Quick Diagnostics

### System Health Check

Run this comprehensive health check to quickly identify common issues:

```bash
#!/bin/bash
# ESG Management System Health Check

echo "=== APG ESG Management Health Check ==="
echo "Date: $(date)"
echo

# Check service status
echo "1. Service Status:"
systemctl is-active esg-management || echo "❌ ESG Management service is not running"
systemctl is-active postgresql || echo "❌ PostgreSQL service is not running"
systemctl is-active redis || echo "❌ Redis service is not running"

# Check database connectivity
echo "2. Database Connectivity:"
python -c "
import psycopg2
try:
    conn = psycopg2.connect('postgresql://esg_user:password@localhost:5432/esg_db')
    print('✅ Database connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"

# Check API health
echo "3. API Health:"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/esg/health | \
grep -q "200" && echo "✅ API is responding" || echo "❌ API is not responding"

# Check disk space
echo "4. Disk Space:"
df -h /opt/apg/capabilities/sustainability_esg_management | awk 'NR==2 {
    if ($5+0 > 90) print "❌ Disk usage is high: " $5
    else print "✅ Disk usage is normal: " $5
}'

# Check memory usage
echo "5. Memory Usage:"
free -h | awk 'NR==2 {
    used = $3+0; total = $2+0; percent = (used/total)*100
    if (percent > 90) print "❌ Memory usage is high: " percent "%"
    else print "✅ Memory usage is normal: " percent "%"
}'

echo
echo "Health check complete. Check individual sections below for specific issues."
```

### Quick Status Commands

```bash
# Service status
sudo systemctl status esg-management
sudo systemctl status postgresql
sudo systemctl status redis

# Log recent errors
sudo journalctl -u esg-management --since "1 hour ago" | grep -i error

# Check API health
curl -H "Content-Type: application/json" http://localhost:8000/api/v1/esg/health

# Database quick check
psql -U esg_user -d esg_db -c "SELECT COUNT(*) as total_metrics FROM esg_metric;"

# Redis connectivity
redis-cli ping
```

---

## Common Issues

### Issue 1: Service Won't Start

**Symptoms:**
- `systemctl start esg-management` fails
- "Failed to start ESG Management service" error

**Diagnosis:**
```bash
# Check detailed status
sudo systemctl status esg-management -l

# Check logs for startup errors
sudo journalctl -u esg-management --no-pager -l

# Check configuration
python manage.py check --deploy
```

**Solutions:**

1. **Configuration Error:**
```bash
# Validate configuration file
python -c "from config.settings import *; print('Configuration valid')"

# Check for missing environment variables
python manage.py check_environment
```

2. **Port Already in Use:**
```bash
# Check what's using port 8000
sudo lsof -i :8000
sudo netstat -tulpn | grep :8000

# Kill conflicting process or change port
sudo kill -9 <PID>
# Or update configuration to use different port
```

3. **Permission Issues:**
```bash
# Fix file permissions
sudo chown -R esg_user:esg_group /opt/apg/capabilities/sustainability_esg_management
sudo chmod -R 755 /opt/apg/capabilities/sustainability_esg_management

# Check user exists
id esg_user
```

### Issue 2: Database Connection Refused

**Symptoms:**
- "Connection refused" errors in logs
- Cannot connect to PostgreSQL

**Diagnosis:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection manually
psql -h localhost -U esg_user -d esg_db -c "SELECT version();"

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log
```

**Solutions:**

1. **PostgreSQL Not Running:**
```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Check startup logs
sudo journalctl -u postgresql --no-pager -l
```

2. **Connection Configuration:**
```bash
# Check pg_hba.conf
sudo cat /etc/postgresql/15/main/pg_hba.conf | grep -v "^#"

# Should include line like:
# host    esg_db    esg_user    127.0.0.1/32    md5

# Reload configuration
sudo systemctl reload postgresql
```

3. **Firewall Issues:**
```bash
# Check if PostgreSQL port is open
sudo ufw status | grep 5432

# If needed, allow internal connections
sudo ufw allow from 127.0.0.1 to any port 5432
```

### Issue 3: High Memory Usage

**Symptoms:**
- System becomes slow
- Out of memory errors
- Application crashes

**Diagnosis:**
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check for memory leaks
python manage.py debug_memory_usage

# Monitor over time
watch -n 5 'free -h && echo && ps aux --sort=-%mem | head -5'
```

**Solutions:**

1. **Optimize Database Connections:**
```python
# In settings.py
DATABASES['default']['CONN_MAX_AGE'] = 0  # Close connections immediately
DATABASES['default']['OPTIONS']['MAX_CONNS'] = 10  # Limit connections
```

2. **Tune Cache Settings:**
```python
# Reduce cache memory usage
CACHES['default']['OPTIONS']['MAX_ENTRIES'] = 1000
CACHES['default']['TIMEOUT'] = 300  # 5 minutes
```

3. **Add Swap Space (Emergency):**
```bash
# Create 2GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Installation Problems

### Problem: APG Capability Registration Fails

**Error Messages:**
- "Failed to register capability with APG platform"
- "Invalid admin token"
- "Capability already exists"

**Solutions:**

1. **Verify APG Platform Connectivity:**
```bash
# Test platform access
curl -H "Authorization: Bearer $APG_ADMIN_TOKEN" \
     $APG_PLATFORM_URL/api/v1/capabilities

# Check DNS resolution
nslookup your-apg-platform.com

# Test network connectivity
telnet your-apg-platform.com 443
```

2. **Token Issues:**
```bash
# Validate admin token
apg auth validate-token $APG_ADMIN_TOKEN

# Generate new token if needed
apg auth login
APG_ADMIN_TOKEN=$(apg auth get-token)
```

3. **Capability Conflicts:**
```bash
# Check existing capabilities
apg capability list | grep sustainability

# Force re-registration
apg capability unregister sustainability_esg_management --force
apg capability register sustainability_esg_management
```

### Problem: Database Migration Failures

**Error Messages:**
- "Migration failed"
- "Table already exists"
- "Column does not exist"

**Solutions:**

1. **Check Migration Status:**
```bash
# Show migration status
python manage.py showmigrations

# Check for conflicts
python manage.py showmigrations --plan
```

2. **Reset Migrations (Development Only):**
```bash
# ⚠️ WARNING: This will delete all data!
python manage.py migrate sustainability_esg_management zero
python manage.py migrate
```

3. **Manual Migration Repair:**
```bash
# Mark specific migration as applied
python manage.py migrate sustainability_esg_management 0001 --fake

# Apply specific migration
python manage.py migrate sustainability_esg_management 0002
```

### Problem: Dependencies Not Found

**Error Messages:**
- "ModuleNotFoundError"
- "ImportError: No module named..."

**Solutions:**

1. **Reinstall Dependencies:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install from requirements
pip install -r requirements.txt

# Check for conflicts
pip check
```

2. **Virtual Environment Issues:**
```bash
# Recreate virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Authentication & Authorization

### Problem: User Cannot Access ESG Features

**Symptoms:**
- "Permission denied" errors
- ESG menu items not visible
- API returns 403 Forbidden

**Diagnosis:**
```bash
# Check user roles
apg auth user-info $USER_ID

# Check ESG permissions
apg auth list-permissions --user $USER_ID | grep esg

# Test specific permission
apg auth check-permission --user $USER_ID --resource esg_metrics --action read
```

**Solutions:**

1. **Assign ESG Roles:**
```bash
# Assign ESG manager role
apg auth assign-role --user $USER_ID --role esg_manager

# Grant specific permissions
apg auth grant-permission --user $USER_ID --resource esg_metrics --action read,create,update
```

2. **Check Tenant Access:**
```bash
# Verify user tenant membership
apg auth tenant-members $TENANT_ID | grep $USER_ID

# Add user to tenant if needed
apg auth add-tenant-member --tenant $TENANT_ID --user $USER_ID
```

### Problem: SSO Integration Issues

**Symptoms:**
- SSO login redirects fail
- "Invalid SAML response" errors
- Users cannot authenticate

**Solutions:**

1. **SAML Configuration:**
```bash
# Check SAML metadata
curl -s $APG_PLATFORM_URL/auth/saml/metadata

# Validate certificate
openssl x509 -in saml_cert.pem -text -noout
```

2. **OAuth Issues:**
```bash
# Test OAuth flow
curl -X POST $APG_PLATFORM_URL/auth/oauth/token \
  -d "grant_type=client_credentials" \
  -d "client_id=$CLIENT_ID" \
  -d "client_secret=$CLIENT_SECRET"
```

---

## Database Issues

### Problem: Database Performance Degradation

**Symptoms:**
- Slow query responses
- Timeouts on dashboard load
- High CPU usage on database server

**Diagnosis:**
```sql
-- Check running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables WHERE schemaname = 'public' ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats WHERE tablename LIKE 'esg_%' ORDER BY n_distinct DESC;
```

**Solutions:**

1. **Create Missing Indexes:**
```sql
-- Metrics table indexes
CREATE INDEX CONCURRENTLY idx_esg_metric_tenant_type_kpi 
ON esg_metric(tenant_id, metric_type, is_kpi) WHERE is_kpi = true;

-- Measurements table indexes
CREATE INDEX CONCURRENTLY idx_esg_measurement_metric_date_desc 
ON esg_measurement(metric_id, measurement_date DESC);

-- Targets table indexes
CREATE INDEX CONCURRENTLY idx_esg_target_tenant_status_date 
ON esg_target(tenant_id, status, target_date) WHERE status != 'completed';
```

2. **Update Table Statistics:**
```sql
-- Analyze tables for better query planning
ANALYZE esg_metric;
ANALYZE esg_measurement;
ANALYZE esg_target;
ANALYZE esg_stakeholder;

-- Check if autovacuum is working
SELECT schemaname, tablename, last_vacuum, last_autovacuum, last_analyze, last_autoanalyze
FROM pg_stat_user_tables WHERE schemaname = 'public';
```

3. **Optimize Configuration:**
```ini
# postgresql.conf optimizations
shared_buffers = 25% of RAM
effective_cache_size = 75% of RAM
work_mem = 256MB
maintenance_work_mem = 1GB
max_connections = 200
```

### Problem: Database Corruption

**Symptoms:**
- "Relation does not exist" errors
- Data inconsistencies
- Checksum failures

**Diagnosis:**
```bash
# Check database integrity
python manage.py check_data_integrity

# PostgreSQL integrity check
psql -d esg_db -c "SELECT datname FROM pg_database WHERE datname = 'esg_db';"

# Check for corruption
psql -d esg_db -c "REINDEX DATABASE esg_db;"
```

**Recovery:**
```bash
# 1. Stop the application
sudo systemctl stop esg-management

# 2. Backup current state
pg_dump -U esg_user esg_db > corrupted_backup_$(date +%Y%m%d_%H%M%S).sql

# 3. Restore from last known good backup
psql -U esg_user -d esg_db < last_good_backup.sql

# 4. Verify data integrity
python manage.py check_data_integrity

# 5. Restart application
sudo systemctl start esg-management
```

---

## API Problems

### Problem: API Timeouts

**Symptoms:**
- Requests timeout after 30 seconds
- "Gateway timeout" errors
- Dashboard loads slowly

**Diagnosis:**
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/v1/esg/metrics"

# Monitor API performance
python manage.py monitor_api_performance --duration 300  # 5 minutes
```

**Solutions:**

1. **Optimize Database Queries:**
```python
# Enable query logging in settings.py
LOGGING['loggers']['django.db.backends'] = {
    'level': 'DEBUG',
    'handlers': ['console'],
}

# Find slow queries
python manage.py analyze_slow_queries
```

2. **Implement Caching:**
```python
# Cache expensive operations
from django.core.cache import cache

def get_dashboard_data(tenant_id):
    cache_key = f"dashboard_data_{tenant_id}"
    data = cache.get(cache_key)
    if not data:
        data = expensive_dashboard_query()
        cache.set(cache_key, data, timeout=300)  # 5 minutes
    return data
```

3. **Add Pagination:**
```python
# Limit large result sets
@api_view(['GET'])
def metrics_list(request):
    queryset = ESGMetric.objects.filter(tenant_id=request.user.tenant_id)
    paginator = Paginator(queryset, 50)  # 50 per page
    page = paginator.get_page(request.GET.get('page', 1))
    return Response(serialize_metrics(page.object_list))
```

### Problem: CORS Issues

**Symptoms:**
- Browser console shows CORS errors
- API requests fail from web interface
- "Access-Control-Allow-Origin" errors

**Solutions:**

1. **Configure CORS Properly:**
```python
# settings.py
CORS_ALLOWED_ORIGINS = [
    "https://your-domain.com",
    "https://esg.your-domain.com",
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'x-tenant-id',
]
```

2. **Check Network Configuration:**
```bash
# Test from command line
curl -H "Origin: https://your-domain.com" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: X-Requested-With" \
     -X OPTIONS \
     http://localhost:8000/api/v1/esg/metrics
```

---

## Performance Issues

### Problem: Slow Dashboard Loading

**Symptoms:**
- Dashboard takes >10 seconds to load
- Browser shows "Loading..." for extended time
- High CPU usage during page load

**Diagnosis:**
```python
# Enable Django debug toolbar
pip install django-debug-toolbar

# Add profiling middleware
MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')

# Profile specific views
python manage.py profile_dashboard_performance
```

**Solutions:**

1. **Optimize Data Fetching:**
```python
# Use select_related and prefetch_related
def get_dashboard_metrics(tenant_id):
    return ESGMetric.objects.filter(tenant_id=tenant_id)\
        .select_related('framework')\
        .prefetch_related('measurements__measurement_date')\
        .only('id', 'name', 'current_value', 'unit', 'is_kpi')
```

2. **Implement Lazy Loading:**
```javascript
// Load dashboard sections progressively
const loadDashboardSection = async (sectionId) => {
    const response = await fetch(`/api/v1/esg/dashboard/${sectionId}`);
    const data = await response.json();
    updateDashboardSection(sectionId, data);
};

// Load sections in parallel
Promise.all([
    loadDashboardSection('metrics'),
    loadDashboardSection('targets'),
    loadDashboardSection('stakeholders')
]);
```

3. **Add Client-Side Caching:**
```javascript
// Cache dashboard data in browser
const dashboardCache = new Map();

const getCachedDashboardData = (key) => {
    const cached = dashboardCache.get(key);
    if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes
        return cached.data;
    }
    return null;
};
```

### Problem: High Memory Usage

**Symptoms:**
- System memory usage >90%
- Application becomes unresponsive
- Out of memory errors

**Solutions:**

1. **Memory Profiling:**
```python
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler manage.py process_esg_data

# Find memory leaks
python manage.py debug_memory_leaks
```

2. **Optimize Data Processing:**
```python
# Process data in chunks
def process_large_dataset(queryset):
    batch_size = 1000
    for batch in queryset.iterator(chunk_size=batch_size):
        process_batch(batch)
        # Force garbage collection
        import gc
        gc.collect()
```

3. **Configure Gunicorn:**
```bash
# gunicorn.conf.py
workers = 2  # Limit number of workers
worker_class = "sync"
worker_connections = 100
max_requests = 1000  # Restart workers periodically
max_requests_jitter = 100
preload_app = True  # Share memory between workers
```

---

## AI/ML Service Issues

### Problem: AI Predictions Unavailable

**Symptoms:**
- "AI service unavailable" errors
- Missing predictions in dashboard
- AI insights not generating

**Diagnosis:**
```bash
# Check AI service status
python manage.py check_ai_service_health

# Test AI endpoint
curl -X POST http://localhost:8000/api/v1/esg/ai/test \
  -H "Content-Type: application/json" \
  -d '{"test": true}'

# Check AI service logs
tail -f /var/log/esg/ai_service.log
```

**Solutions:**

1. **Restart AI Service:**
```bash
# Restart AI orchestration service
sudo systemctl restart ai-orchestration

# Check service dependencies
python manage.py check_ai_dependencies
```

2. **Model Issues:**
```python
# Reload AI models
python manage.py reload_ai_models

# Test model predictions
python manage.py test_prediction_models --model environmental_lstm
```

3. **Fallback to Basic Analytics:**
```python
# Disable AI features temporarily
ESG_SETTINGS['AI_ENABLED'] = False

# Use statistical methods instead
def fallback_prediction(metric_data):
    # Simple moving average prediction
    recent_values = metric_data[-12:]  # Last 12 measurements
    return sum(recent_values) / len(recent_values)
```

### Problem: Poor Prediction Accuracy

**Symptoms:**
- AI predictions significantly off target
- Low confidence scores
- Stakeholders complaining about accuracy

**Solutions:**

1. **Retrain Models:**
```python
# Trigger model retraining
python manage.py retrain_ai_models --force

# Update training data
python manage.py update_training_dataset --include-recent-data
```

2. **Adjust Model Parameters:**
```python
# Increase training data window
AI_ORCHESTRATION_CONFIG['TRAINING_WINDOW_MONTHS'] = 24

# Lower confidence threshold
AI_ORCHESTRATION_CONFIG['CONFIDENCE_THRESHOLD'] = 0.6
```

---

## Real-time Features

### Problem: WebSocket Connections Failing

**Symptoms:**
- Real-time updates not working
- WebSocket connection errors in browser
- "Connection refused" errors

**Diagnosis:**
```bash
# Test WebSocket endpoint
wscat -c ws://localhost:8000/ws/esg/

# Check WebSocket logs
grep -i websocket /var/log/esg/esg.log

# Test with browser developer tools
# Open browser console and run:
# const ws = new WebSocket('ws://localhost:8000/ws/esg/');
# ws.onopen = () => console.log('Connected');
# ws.onerror = (error) => console.log('Error:', error);
```

**Solutions:**

1. **Configure WebSocket Support:**
```python
# settings.py - ensure channels is configured
ASGI_APPLICATION = 'sustainability_esg_management.asgi.application'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [('127.0.0.1', 6379)],
        },
    },
}
```

2. **Proxy Configuration:**
```nginx
# nginx configuration for WebSocket support
location /ws/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 86400;
}
```

### Problem: Server-Sent Events Not Working

**Symptoms:**
- Live dashboard updates not working
- Event stream connection drops
- Browser shows connection errors

**Solutions:**

1. **Check Event Stream Configuration:**
```python
# views.py - SSE endpoint
@api_view(['GET'])
def event_stream(request):
    def event_generator():
        while True:
            # Get latest updates
            updates = get_latest_esg_updates(request.user.tenant_id)
            for update in updates:
                yield f"data: {json.dumps(update)}\n\n"
            time.sleep(5)  # 5-second interval
    
    response = StreamingHttpResponse(
        event_generator(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['Connection'] = 'keep-alive'
    return response
```

2. **Browser-Side Event Handling:**
```javascript
// Robust EventSource implementation
const connectEventSource = () => {
    const eventSource = new EventSource('/api/v1/esg/events');
    
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
    
    eventSource.onerror = (error) => {
        console.log('EventSource failed:', error);
        eventSource.close();
        // Reconnect after 5 seconds
        setTimeout(connectEventSource, 5000);
    };
    
    return eventSource;
};
```

---

## Dashboard & UI Problems

### Problem: Dashboard Not Loading

**Symptoms:**
- Blank dashboard page
- JavaScript errors in browser console
- CSS/styling issues

**Diagnosis:**
```bash
# Check static files
python manage.py collectstatic --dry-run

# Verify file permissions
ls -la /opt/apg/capabilities/sustainability_esg_management/static/

# Check for JavaScript errors
# Open browser developer tools (F12) and check Console tab
```

**Solutions:**

1. **Static Files Issues:**
```bash
# Collect static files
python manage.py collectstatic --clear --noinput

# Fix permissions
sudo chown -R www-data:www-data /opt/apg/capabilities/sustainability_esg_management/static/
```

2. **Template Issues:**
```python
# Check template loading
python manage.py check_templates

# Debug template rendering
TEMPLATES[0]['OPTIONS']['debug'] = True
```

### Problem: Data Not Displaying

**Symptoms:**
- Dashboard loads but shows no data
- "No data available" messages
- Correct data exists in database

**Solutions:**

1. **Check Data Permissions:**
```python
# Debug data filtering
def debug_dashboard_data(request):
    tenant_id = request.user.tenant_id
    metrics = ESGMetric.objects.filter(tenant_id=tenant_id)
    print(f"Found {metrics.count()} metrics for tenant {tenant_id}")
    return metrics
```

2. **Verify API Endpoints:**
```bash
# Test API directly
curl -H "Authorization: Bearer $TOKEN" \
     -H "X-Tenant-ID: $TENANT_ID" \
     "http://localhost:8000/api/v1/esg/metrics?limit=5"
```

---

## Data Import/Export Issues

### Problem: CSV Import Failures

**Symptoms:**
- "Invalid file format" errors
- Import process hangs
- Partial data import

**Solutions:**

1. **Validate CSV Format:**
```python
# Check CSV structure
import csv
import io

def validate_csv_format(file_content):
    reader = csv.DictReader(io.StringIO(file_content))
    required_fields = ['name', 'code', 'metric_type', 'unit']
    
    if not all(field in reader.fieldnames for field in required_fields):
        missing = [f for f in required_fields if f not in reader.fieldnames]
        raise ValueError(f"Missing required fields: {missing}")
    
    return True
```

2. **Handle Large Files:**
```python
# Process large CSV files in chunks
def import_large_csv(file_path, chunk_size=1000):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        chunk = []
        
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                process_chunk(chunk)
                chunk = []
        
        if chunk:  # Process remaining rows
            process_chunk(chunk)
```

### Problem: Report Generation Failures

**Symptoms:**
- Report generation times out
- Incomplete reports
- "Memory error" during generation

**Solutions:**

1. **Optimize Report Generation:**
```python
# Generate reports in background
from celery import shared_task

@shared_task
def generate_report_async(report_id, tenant_id):
    try:
        report = ESGReport.objects.get(id=report_id)
        report.status = 'generating'
        report.save()
        
        # Generate report content
        content = generate_report_content(tenant_id, report.parameters)
        
        # Save to file
        report.file_path = save_report_file(content, report.name)
        report.status = 'completed'
        report.save()
        
    except Exception as e:
        report.status = 'failed'
        report.error_message = str(e)
        report.save()
```

2. **Memory Management:**
```python
# Use streaming for large reports
def generate_streaming_report(data_query):
    def report_generator():
        yield report_header()
        
        for chunk in data_query.iterator(chunk_size=1000):
            yield process_report_chunk(chunk)
            
        yield report_footer()
    
    return report_generator()
```

---

## Integration Problems

### Problem: External API Integration Failures

**Symptoms:**
- "API key invalid" errors
- Connection timeouts to external services
- Data sync failures

**Solutions:**

1. **Validate API Credentials:**
```python
# Test external API connectivity
def test_external_api(api_name, api_key):
    apis = {
        'weather': 'https://api.weatherapi.com/v1/current.json',
        'emissions': 'https://api.epa.gov/easiur/rest/getFacilityInfo'
    }
    
    url = apis.get(api_name)
    response = requests.get(url, params={'key': api_key, 'q': 'test'})
    
    if response.status_code == 200:
        print(f"✅ {api_name} API connection successful")
    else:
        print(f"❌ {api_name} API failed: {response.status_code}")
```

2. **Implement Retry Logic:**
```python
# Robust API calls with retries
import time
from functools import wraps

def retry_api_call(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry_api_call(max_retries=3)
def fetch_external_data(api_url, params):
    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()
```

---

## Error Code Reference

### ESG-1000 Series: General Errors

| Code | Message | Cause | Solution |
|------|---------|-------|----------|
| ESG-1001 | Service unavailable | Service not running | Start esg-management service |
| ESG-1002 | Configuration error | Invalid config file | Check settings.py syntax |
| ESG-1003 | Database connection failed | DB not accessible | Check database status |

### ESG-2000 Series: Authentication Errors

| Code | Message | Cause | Solution |
|------|---------|-------|----------|
| ESG-2001 | Invalid token | Expired JWT token | Refresh authentication token |
| ESG-2002 | Permission denied | Insufficient privileges | Check user roles |
| ESG-2003 | Tenant not found | Invalid tenant ID | Verify tenant configuration |

### ESG-3000 Series: Data Errors

| Code | Message | Cause | Solution |
|------|---------|-------|----------|
| ESG-3001 | Metric not found | Invalid metric ID | Check metric exists |
| ESG-3002 | Validation failed | Invalid data format | Check input validation |
| ESG-3003 | Duplicate entry | Unique constraint violation | Check for existing records |

### ESG-4000 Series: AI/ML Errors

| Code | Message | Cause | Solution |
|------|---------|-------|----------|
| ESG-4001 | AI service unavailable | AI service down | Restart ai-orchestration |
| ESG-4002 | Model not found | Missing ML model | Reload AI models |
| ESG-4003 | Prediction failed | Model error | Check model parameters |

### ESG-5000 Series: Integration Errors

| Code | Message | Cause | Solution |
|------|---------|-------|----------|
| ESG-5001 | External API failed | Third-party service issue | Check API status |
| ESG-5002 | Import failed | File format error | Validate file format |
| ESG-5003 | Export timeout | Large dataset | Use background processing |

---

## Log Analysis

### Log Locations

```bash
# Application logs
/opt/apg/capabilities/sustainability_esg_management/logs/esg.log
/opt/apg/capabilities/sustainability_esg_management/logs/ai_service.log
/opt/apg/capabilities/sustainability_esg_management/logs/celery.log

# System logs
/var/log/syslog
/var/log/postgresql/postgresql-15-main.log
/var/log/redis/redis-server.log

# Systemd logs
journalctl -u esg-management
journalctl -u postgresql
journalctl -u redis
```

### Log Analysis Commands

```bash
# Find recent errors
tail -f /opt/apg/capabilities/sustainability_esg_management/logs/esg.log | grep -i error

# Search for specific issues
grep -i "database" /opt/apg/capabilities/sustainability_esg_management/logs/esg.log | tail -20

# Analyze error patterns
awk '/ERROR/ {print $1, $2, $5}' /opt/apg/capabilities/sustainability_esg_management/logs/esg.log | sort | uniq -c | sort -nr

# Monitor API performance
grep "api/v1/esg" /opt/apg/capabilities/sustainability_esg_management/logs/esg.log | grep -E "[0-9]+ms" | tail -10
```

### Log Rotation Configuration

```bash
# /etc/logrotate.d/esg-management
/opt/apg/capabilities/sustainability_esg_management/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 esg_user esg_group
    postrotate
        systemctl reload esg-management
    endscript
}
```

---

## Recovery Procedures

### Emergency Recovery Steps

1. **Service Recovery:**
```bash
#!/bin/bash
# Emergency service recovery script

echo "Starting emergency recovery..."

# Stop all services
sudo systemctl stop esg-management
sudo systemctl stop redis
sudo systemctl stop postgresql

# Check disk space
if [ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -gt 95 ]; then
    echo "Critical: Disk space >95%. Cleaning logs..."
    find /var/log -name "*.log" -mtime +7 -delete
    find /opt/apg/capabilities/sustainability_esg_management/logs -name "*.log.gz" -mtime +30 -delete
fi

# Start services in order
sudo systemctl start postgresql
sleep 10
sudo systemctl start redis  
sleep 5
sudo systemctl start esg-management

# Verify services
for service in postgresql redis esg-management; do
    if systemctl is-active --quiet $service; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
        journalctl -u $service --no-pager -l | tail -10
    fi
done

echo "Recovery complete. Check service status manually."
```

2. **Database Recovery:**
```bash
#!/bin/bash
# Database recovery script

echo "Starting database recovery..."

# Create emergency backup
pg_dump -U esg_user esg_db > emergency_backup_$(date +%Y%m%d_%H%M%S).sql

# Check database integrity
psql -U esg_user -d esg_db -c "SELECT pg_database_size('esg_db');"

# Repair corrupted indexes
psql -U esg_user -d esg_db -c "REINDEX DATABASE esg_db;"

# Update statistics
psql -U esg_user -d esg_db -c "ANALYZE;"

echo "Database recovery complete."
```

### Data Recovery

```python
# Recover deleted data from audit logs
def recover_deleted_record(table_name, record_id, delete_timestamp):
    """
    Recover deleted record from audit trail
    """
    from sustainability_esg_management.models import AuditLog
    
    # Find the delete operation
    delete_log = AuditLog.objects.filter(
        table_name=table_name,
        record_id=record_id,
        action='DELETE',
        timestamp__gte=delete_timestamp - timedelta(minutes=5),
        timestamp__lte=delete_timestamp + timedelta(minutes=5)
    ).first()
    
    if delete_log and delete_log.old_values:
        # Restore the record
        model_class = apps.get_model('sustainability_esg_management', table_name)
        restored_record = model_class(**delete_log.old_values)
        restored_record.save()
        
        return restored_record
    
    return None
```

---

## Support Escalation

### When to Escalate

Escalate to higher support tiers when:

1. **Service down >30 minutes** and basic troubleshooting fails
2. **Data corruption** or loss detected
3. **Security incident** suspected
4. **Performance degradation >50%** affecting multiple users
5. **Multiple system failures** occurring simultaneously

### Support Levels

#### Level 1 - Basic Support
- **Response Time:** 4 hours (business hours)
- **Scope:** Common configuration issues, basic troubleshooting
- **Contact:** support@datacraft.co.ke

#### Level 2 - Advanced Support  
- **Response Time:** 2 hours (business hours), 8 hours (24/7)
- **Scope:** Performance issues, integration problems, data recovery
- **Contact:** advanced-support@datacraft.co.ke

#### Level 3 - Emergency Support
- **Response Time:** 30 minutes (24/7)
- **Scope:** Critical system failures, security incidents, data loss
- **Contact:** emergency@datacraft.co.ke, +254-XXX-XXXX

### Information to Provide

When contacting support, include:

1. **System Information:**
```bash
# Generate support package
python manage.py generate_support_package --include-logs --include-config

# System details
uname -a
python --version
psql --version
redis-server --version
```

2. **Error Details:**
- Exact error messages
- Time when issue started
- Steps to reproduce
- Recent changes made

3. **Log Excerpts:**
```bash
# Last 100 lines of main log
tail -100 /opt/apg/capabilities/sustainability_esg_management/logs/esg.log

# Database errors
grep -i error /var/log/postgresql/postgresql-15-main.log | tail -20
```

### Remote Support Access

For complex issues, support may request remote access:

```bash
# Install support tools (if approved)
sudo apt-get install openssh-server screen

# Create temporary support user (24-hour access)
sudo useradd -m -s /bin/bash esg_support
sudo usermod -aG sudo esg_support
sudo passwd esg_support  # Use provided temporary password

# Enable SSH access (if required)
sudo systemctl start ssh
sudo systemctl enable ssh

# Remember to remove access after issue resolution:
sudo deluser --remove-home esg_support
```

---

## Preventive Measures

### Regular Health Checks

```bash
#!/bin/bash
# Weekly health check script (add to cron)

# Check disk space
df -h | awk '$5 > 80 {print "Warning: " $1 " is " $5 " full"}'

# Check database size
psql -U esg_user -d esg_db -c "
SELECT pg_size_pretty(pg_database_size('esg_db')) as size;
"

# Check for long-running queries
psql -U esg_user -d esg_db -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '1 minute'
AND state = 'active';
"

# Check log file sizes
find /opt/apg/capabilities/sustainability_esg_management/logs -name "*.log" -size +100M

# Test API health
curl -f http://localhost:8000/api/v1/esg/health || echo "API health check failed"
```

### Monitoring Setup

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: esg_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: esg_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards

volumes:
  prometheus_data:
  grafana_data:
```

---

**Copyright © 2025 Datacraft - All rights reserved.**  
**Author: Nyimbi Odero <nyimbi@gmail.com>**  
**Website: www.datacraft.co.ke**

---

**🔧 Need Additional Help?**

- **Emergency Support:** emergency@datacraft.co.ke
- **Community Forum:** https://community.apg.platform/sustainability-esg
- **Knowledge Base:** https://docs.datacraft.co.ke/esg-management
- **GitHub Issues:** https://github.com/apg-platform/sustainability-esg-management/issues