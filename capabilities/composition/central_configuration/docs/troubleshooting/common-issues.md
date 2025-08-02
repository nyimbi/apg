# Troubleshooting Guide

## Common Issues and Solutions

### Database Connection Issues

#### Issue: Connection Refused
```
Error: connection to server at "localhost" (127.0.0.1), port 5432 failed: Connection refused
```

**Causes:**
- PostgreSQL service not running
- Incorrect connection parameters
- Firewall blocking connections
- Database server overloaded

**Solutions:**

1. **Check PostgreSQL Service Status:**
   ```bash
   # Linux/macOS
   sudo systemctl status postgresql
   # or
   brew services list | grep postgresql
   
   # Start if not running
   sudo systemctl start postgresql
   # or
   brew services start postgresql
   ```

2. **Verify Connection Parameters:**
   ```python
   # Test connection manually
   import asyncpg
   import asyncio
   
   async def test_connection():
       try:
           conn = await asyncpg.connect(
               host="localhost",
               port=5432,
               user="config_user",
               password="your_password",
               database="central_config"
           )
           print("Connection successful!")
           await conn.close()
       except Exception as e:
           print(f"Connection failed: {e}")
   
   asyncio.run(test_connection())
   ```

3. **Check Firewall Settings:**
   ```bash
   # Allow PostgreSQL port
   sudo ufw allow 5432/tcp
   
   # Check if port is listening
   netstat -tlnp | grep 5432
   ```

4. **Increase Connection Pool:**
   ```python
   # In your configuration
   DATABASE_CONFIG = {
       "min_pool_size": 5,
       "max_pool_size": 50,
       "max_inactive_connection_lifetime": 300,
       "timeout": 60
   }
   ```

#### Issue: Connection Pool Exhausted
```
Error: asyncpg.exceptions.TooManyConnectionsError: too many connections for role "config_user"
```

**Solutions:**

1. **Increase PostgreSQL max_connections:**
   ```sql
   -- Check current connections
   SELECT count(*) FROM pg_stat_activity;
   
   -- Show max connections
   SHOW max_connections;
   
   -- Increase max connections (requires restart)
   ALTER SYSTEM SET max_connections = 200;
   SELECT pg_reload_conf();
   ```

2. **Optimize Connection Pool Settings:**
   ```python
   DATABASE_CONFIG = {
       "min_pool_size": 2,
       "max_pool_size": 20,  # Reduced from default
       "max_inactive_connection_lifetime": 60,  # Close idle connections faster
       "command_timeout": 30
   }
   ```

3. **Implement Connection Retry Logic:**
   ```python
   import asyncio
   from asyncpg.exceptions import TooManyConnectionsError
   
   async def get_connection_with_retry(pool, max_retries=3):
       for attempt in range(max_retries):
           try:
               return await pool.acquire()
           except TooManyConnectionsError:
               if attempt < max_retries - 1:
                   await asyncio.sleep(2 ** attempt)  # Exponential backoff
               else:
                   raise
   ```

### Redis Connection Issues

#### Issue: Redis Connection Timeout
```
Error: redis.exceptions.TimeoutError: Timeout connecting to server
```

**Solutions:**

1. **Check Redis Service:**
   ```bash
   # Check Redis status
   redis-cli ping
   # Should return PONG
   
   # Check Redis logs
   tail -f /var/log/redis/redis-server.log
   ```

2. **Verify Redis Configuration:**
   ```python
   import redis.asyncio as redis
   
   # Test Redis connection
   async def test_redis():
       try:
           r = redis.Redis(
               host="localhost",
               port=6379,
               db=0,
               socket_timeout=5,
               socket_connect_timeout=5,
               health_check_interval=30
           )
           await r.ping()
           print("Redis connection successful!")
           await r.close()
       except Exception as e:
           print(f"Redis connection failed: {e}")
   ```

3. **Increase Timeout Settings:**
   ```python
   REDIS_CONFIG = {
       "socket_timeout": 10,
       "socket_connect_timeout": 10,
       "socket_keepalive": True,
       "socket_keepalive_options": {},
       "health_check_interval": 30,
       "retry_on_timeout": True,
       "max_connections": 100
   }
   ```

### Encryption/Decryption Errors

#### Issue: Quantum Crypto Library Not Found
```
Error: liboqs library not found. Install with: pip install liboqs-python
```

**Solutions:**

1. **Install Quantum-Resistant Crypto Libraries:**
   ```bash
   # Install liboqs
   pip install liboqs-python
   
   # On Ubuntu/Debian
   sudo apt-get install liboqs-dev
   
   # On macOS
   brew install liboqs
   
   # Verify installation
   python -c "import oqs; print('liboqs installed successfully')"
   ```

2. **Fallback to Classical Encryption:**
   ```python
   # In security_engine.py
   try:
       import oqs
       QUANTUM_CRYPTO_AVAILABLE = True
   except ImportError:
       QUANTUM_CRYPTO_AVAILABLE = False
       logger.warning("Quantum crypto not available, falling back to classical encryption")
   
   # Use fallback encryption
   if not QUANTUM_CRYPTO_AVAILABLE:
       from cryptography.fernet import Fernet
       # Implement classical encryption fallback
   ```

#### Issue: Key Decryption Failed
```
Error: Failed to decrypt configuration value: Invalid key or corrupted data
```

**Solutions:**

1. **Check Key Rotation Status:**
   ```python
   # Verify key information
   key_info = await security_engine.get_key_info(tenant_id="your_tenant")
   print(f"Current key version: {key_info['version']}")
   print(f"Key rotation date: {key_info['rotation_date']}")
   ```

2. **Implement Multi-Version Key Support:**
   ```python
   async def decrypt_with_fallback(encrypted_data, tenant_id):
       """Try decryption with current and previous keys"""
       current_key = await get_current_key(tenant_id)
       
       try:
           return await decrypt_data(encrypted_data, current_key)
       except DecryptionError:
           # Try previous key versions
           for i in range(1, 4):  # Try last 3 key versions
               try:
                   old_key = await get_historical_key(tenant_id, versions_back=i)
                   if old_key:
                       return await decrypt_data(encrypted_data, old_key)
               except DecryptionError:
                   continue
           raise DecryptionError("Unable to decrypt with any available keys")
   ```

3. **Validate Key Integrity:**
   ```bash
   # Check key file permissions
   ls -la /etc/apg/keys/
   
   # Verify key file checksums
   sha256sum /etc/apg/keys/*
   ```

### Real-time Synchronization Issues

#### Issue: WebSocket Connection Drops
```
Error: websockets.exceptions.ConnectionClosed: received 1006 (abnormal closure)
```

**Solutions:**

1. **Implement Connection Retry Logic:**
   ```javascript
   class RobustWebSocketClient {
       constructor(url) {
           this.url = url;
           this.reconnectAttempts = 0;
           this.maxReconnectAttempts = 5;
           this.reconnectInterval = 1000;
       }
       
       connect() {
           this.ws = new WebSocket(this.url);
           
           this.ws.onopen = () => {
               console.log('WebSocket connected');
               this.reconnectAttempts = 0;
           };
           
           this.ws.onclose = (event) => {
               if (event.code !== 1000) {  // Not normal closure
                   this.handleReconnection();
               }
           };
           
           this.ws.onerror = (error) => {
               console.error('WebSocket error:', error);
           };
       }
       
       handleReconnection() {
           if (this.reconnectAttempts < this.maxReconnectAttempts) {
               this.reconnectAttempts++;
               const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
               
               setTimeout(() => {
                   console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
                   this.connect();
               }, delay);
           }
       }
   }
   ```

2. **Configure WebSocket Keepalive:**
   ```python
   # Server-side WebSocket configuration
   websocket_config = {
       "ping_interval": 30,  # Send ping every 30 seconds
       "ping_timeout": 10,   # Wait 10 seconds for pong
       "close_timeout": 10,  # Wait 10 seconds for close
       "max_size": 2**20,    # 1MB max message size
       "max_queue": 32       # Max queued messages
   }
   ```

3. **Monitor Connection Health:**
   ```python
   async def monitor_websocket_health():
       """Monitor WebSocket connection health"""
       while True:
           await asyncio.sleep(60)  # Check every minute
           
           dead_connections = []
           for conn_id, websocket in websocket_connections.items():
               try:
                   pong_waiter = await websocket.ping()
                   await asyncio.wait_for(pong_waiter, timeout=5)
               except Exception:
                   dead_connections.append(conn_id)
           
           # Clean up dead connections
           for conn_id in dead_connections:
               await remove_websocket_connection(conn_id)
   ```

#### Issue: Kafka Consumer Lag
```
Warning: Kafka consumer lag detected: 5000 messages behind
```

**Solutions:**

1. **Increase Consumer Parallelism:**
   ```python
   # Configure multiple consumer instances
   consumer_config = {
       "group_id": f"apg_config_consumer_{node_id}",
       "max_poll_records": 500,
       "fetch_min_bytes": 1024,
       "fetch_max_wait_ms": 500,
       "session_timeout_ms": 30000,
       "heartbeat_interval_ms": 3000,
       "enable_auto_commit": True,
       "auto_commit_interval_ms": 1000
   }
   
   # Start multiple consumer instances
   for i in range(3):  # 3 consumer instances
       consumer = AIOKafkaConsumer(**consumer_config)
       asyncio.create_task(run_consumer(consumer))
   ```

2. **Optimize Message Processing:**
   ```python
   async def process_messages_batch(messages):
       """Process messages in batches for better throughput"""
       batch_size = 100
       
       for i in range(0, len(messages), batch_size):
           batch = messages[i:i + batch_size]
           
           # Process batch concurrently
           tasks = [process_single_message(msg) for msg in batch]
           await asyncio.gather(*tasks, return_exceptions=True)
   ```

3. **Monitor and Alert on Lag:**
   ```python
   async def monitor_consumer_lag():
       """Monitor Kafka consumer lag"""
       while True:
           try:
               # Get consumer group info
               group_info = await kafka_admin.describe_consumer_groups([group_id])
               
               for topic_partition, offset_metadata in group_info.items():
                   lag = offset_metadata.high_water_mark - offset_metadata.committed_offset
                   
                   if lag > 1000:  # Alert if lag > 1000 messages
                       await send_alert(f"High consumer lag: {lag} messages")
               
               await asyncio.sleep(60)  # Check every minute
               
           except Exception as e:
               logger.error(f"Error monitoring consumer lag: {e}")
   ```

### Performance Issues

#### Issue: Slow Configuration Retrieval
```
Warning: Configuration lookup took 2.5 seconds, expected < 100ms
```

**Solutions:**

1. **Implement Redis Caching:**
   ```python
   async def get_config_with_cache(key, tenant_id):
       """Get configuration with Redis caching"""
       cache_key = f"config:{tenant_id}:{key}"
       
       # Try cache first
       cached_value = await redis_client.get(cache_key)
       if cached_value:
           return json.loads(cached_value)
       
       # Cache miss - get from database
       value = await get_config_from_db(key, tenant_id)
       
       # Cache for 5 minutes
       await redis_client.setex(
           cache_key, 
           300, 
           json.dumps(value)
       )
       
       return value
   ```

2. **Add Database Indexing:**
   ```sql
   -- Add indexes for common queries
   CREATE INDEX CONCURRENTLY idx_config_tenant_key 
   ON configurations(tenant_id, config_key);
   
   CREATE INDEX CONCURRENTLY idx_config_updated_at 
   ON configurations(updated_at DESC);
   
   CREATE INDEX CONCURRENTLY idx_config_user_id 
   ON configurations(user_id) 
   WHERE user_id IS NOT NULL;
   
   -- Analyze index usage
   SELECT schemaname, tablename, attname, n_distinct, correlation
   FROM pg_stats 
   WHERE tablename = 'configurations';
   ```

3. **Implement Query Optimization:**
   ```python
   # Use connection pooling and prepared statements
   async def get_configs_optimized(keys, tenant_id):
       """Optimized batch configuration retrieval"""
       
       # Use prepared statement for better performance
       query = """
       SELECT config_key, config_value, updated_at
       FROM configurations 
       WHERE tenant_id = $1 AND config_key = ANY($2)
       """
       
       async with db_pool.acquire() as conn:
           rows = await conn.fetch(query, tenant_id, keys)
           
       return {row['config_key']: row['config_value'] for row in rows}
   ```

#### Issue: High Memory Usage
```
Warning: Memory usage at 85%, triggering garbage collection
```

**Solutions:**

1. **Implement Memory Monitoring:**
   ```python
   import psutil
   import gc
   
   async def monitor_memory_usage():
       """Monitor and manage memory usage"""
       while True:
           process = psutil.Process()
           memory_percent = process.memory_percent()
           
           if memory_percent > 80:
               logger.warning(f"High memory usage: {memory_percent:.1f}%")
               
               # Force garbage collection
               gc.collect()
               
               # Clear caches if needed
               await clear_expired_caches()
           
           await asyncio.sleep(30)
   ```

2. **Optimize Caching Strategy:**
   ```python
   # Implement LRU cache with size limits
   from cachetools import TTLCache
   
   class OptimizedConfigCache:
       def __init__(self, max_size=10000, ttl=300):
           self.cache = TTLCache(maxsize=max_size, ttl=ttl)
           self.hit_count = 0
           self.miss_count = 0
       
       async def get(self, key):
           if key in self.cache:
               self.hit_count += 1
               return self.cache[key]
           
           self.miss_count += 1
           return None
       
       async def set(self, key, value):
           self.cache[key] = value
       
       def get_hit_rate(self):
           total = self.hit_count + self.miss_count
           return self.hit_count / total if total > 0 else 0
   ```

3. **Profile Memory Usage:**
   ```bash
   # Use memory profiler
   pip install memory-profiler
   
   # Run with memory profiling
   python -m memory_profiler your_script.py
   
   # Monitor memory usage over time
   mprof run your_script.py
   mprof plot
   ```

### Security Issues

#### Issue: Authentication Failures
```
Error: JWT token validation failed: Token has expired
```

**Solutions:**

1. **Implement Token Refresh:**
   ```python
   async def refresh_jwt_token(current_token):
       """Refresh JWT token before expiration"""
       try:
           # Decode without verification to check expiry
           payload = jwt.decode(current_token, options={"verify_signature": False})
           exp_timestamp = payload.get('exp', 0)
           
           # Refresh if token expires in less than 5 minutes
           if exp_timestamp - time.time() < 300:
               new_token = await request_token_refresh(current_token)
               return new_token
           
           return current_token
           
       except jwt.InvalidTokenError:
           # Token is invalid, need to re-authenticate
           return await authenticate_user()
   ```

2. **Add Token Validation Middleware:**
   ```python
   from functools import wraps
   
   def require_valid_token(f):
       @wraps(f)
       async def decorated_function(*args, **kwargs):
           token = request.headers.get('Authorization', '').replace('Bearer ', '')
           
           try:
               payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
               request.user_id = payload.get('user_id')
               request.tenant_id = payload.get('tenant_id')
               
           except jwt.ExpiredSignatureError:
               return jsonify({'error': 'Token has expired'}), 401
           except jwt.InvalidTokenError:
               return jsonify({'error': 'Invalid token'}), 401
           
           return await f(*args, **kwargs)
       
       return decorated_function
   ```

#### Issue: Rate Limiting Triggered
```
Error: Rate limit exceeded: 100 requests per minute
```

**Solutions:**

1. **Implement Exponential Backoff:**
   ```python
   import asyncio
   from random import uniform
   
   async def make_request_with_backoff(request_func, max_retries=5):
       """Make request with exponential backoff on rate limit"""
       
       for attempt in range(max_retries):
           try:
               return await request_func()
               
           except RateLimitError as e:
               if attempt < max_retries - 1:
                   # Extract retry-after from error if available
                   retry_after = getattr(e, 'retry_after', None)
                   if retry_after:
                       delay = float(retry_after)
                   else:
                       delay = (2 ** attempt) + uniform(0, 1)
                   
                   logger.warning(f"Rate limited, retrying in {delay:.1f}s")
                   await asyncio.sleep(delay)
               else:
                   raise
   ```

2. **Implement Client-Side Rate Limiting:**
   ```python
   import asyncio
   from collections import deque
   from time import time
   
   class RateLimiter:
       def __init__(self, max_requests=100, window_seconds=60):
           self.max_requests = max_requests
           self.window_seconds = window_seconds
           self.requests = deque()
       
       async def acquire(self):
           """Acquire rate limit permission"""
           current_time = time()
           
           # Remove old requests outside the window
           while self.requests and self.requests[0] <= current_time - self.window_seconds:
               self.requests.popleft()
           
           # Check if we can make another request
           if len(self.requests) >= self.max_requests:
               # Calculate wait time
               oldest_request = self.requests[0]
               wait_time = self.window_seconds - (current_time - oldest_request)
               await asyncio.sleep(wait_time)
               return await self.acquire()  # Recursive call
           
           # Add current request
           self.requests.append(current_time)
   ```

### ML Model Issues

#### Issue: Model Prediction Errors
```
Error: Model prediction failed: Input shape mismatch
```

**Solutions:**

1. **Implement Input Validation:**
   ```python
   import numpy as np
   from typing import Dict, Any
   
   def validate_model_input(features: Dict[str, Any], expected_schema: Dict[str, Any]) -> Dict[str, Any]:
       """Validate and transform model input features"""
       validated_features = {}
       
       for feature_name, expected_type in expected_schema.items():
           if feature_name not in features:
               # Use default value or raise error
               if 'default' in expected_type:
                   validated_features[feature_name] = expected_type['default']
               else:
                   raise ValueError(f"Missing required feature: {feature_name}")
           else:
               value = features[feature_name]
               
               # Type validation and conversion
               if expected_type['type'] == 'float':
                   validated_features[feature_name] = float(value)
               elif expected_type['type'] == 'int':
                   validated_features[feature_name] = int(value)
               elif expected_type['type'] == 'categorical':
                   if value not in expected_type['categories']:
                       raise ValueError(f"Invalid category for {feature_name}: {value}")
                   validated_features[feature_name] = value
       
       return validated_features
   ```

2. **Add Model Fallback Logic:**
   ```python
   async def robust_model_prediction(model_name, features, fallback_models=None):
       """Make prediction with fallback models"""
       fallback_models = fallback_models or []
       
       # Try primary model
       try:
           return await ml_engine.predict(model_name, features)
       except ModelError as e:
           logger.warning(f"Primary model {model_name} failed: {e}")
           
           # Try fallback models
           for fallback_model in fallback_models:
               try:
                   result = await ml_engine.predict(fallback_model, features)
                   logger.info(f"Using fallback model {fallback_model}")
                   return result
               except ModelError:
                   continue
           
           # All models failed, use rule-based fallback
           return get_rule_based_prediction(features)
   ```

#### Issue: Model Drift Detected
```
Warning: Model performance degraded: F1 score dropped from 0.89 to 0.72
```

**Solutions:**

1. **Automated Model Retraining:**
   ```python
   async def handle_model_drift(model_name, drift_metrics):
       """Handle detected model drift"""
       logger.warning(f"Model drift detected for {model_name}: {drift_metrics}")
       
       # Check if drift is significant
       if drift_metrics['performance_drop'] > 0.1:  # 10% performance drop
           # Trigger automatic retraining
           retrain_job = await ml_engine.start_model_retraining(
               model_name,
               training_data_query={
                   "date_range": "30_days",  # Use recent data
                   "include_drift_samples": True
               },
               priority="high"
           )
           
           # Notify team
           await send_notification(
               "model_drift",
               f"Model {model_name} is being retrained due to performance drift",
               priority="high"
           )
           
           return retrain_job
   ```

2. **A/B Test New Model:**
   ```python
   async def deploy_model_with_ab_test(new_model_version, current_model_version):
       """Deploy new model with A/B testing"""
       ab_test_config = {
           "control": {
               "model_version": current_model_version,
               "traffic_percentage": 80
           },
           "treatment": {
               "model_version": new_model_version,
               "traffic_percentage": 20
           },
           "success_metrics": ["accuracy", "f1_score", "latency"],
           "duration_days": 7,
           "auto_promote": True,
           "promotion_threshold": 0.05  # 5% improvement
       }
       
       return await ml_engine.start_ab_test(ab_test_config)
   ```

## Diagnostic Tools

### Health Check Script

```python
#!/usr/bin/env python3
"""
APG Central Configuration Health Check Script
"""

import asyncio
import asyncpg
import redis.asyncio as redis
import json
import time
from datetime import datetime

class HealthChecker:
    def __init__(self):
        self.results = {}
    
    async def check_database(self):
        """Check PostgreSQL database connectivity and performance"""
        try:
            start_time = time.time()
            
            conn = await asyncpg.connect(
                host="localhost",
                port=5432,
                user="config_user",
                password="password",
                database="central_config"
            )
            
            # Test query
            await conn.fetchval("SELECT 1")
            
            # Check table exists
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            await conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            self.results['database'] = {
                'status': 'healthy',
                'response_time_ms': round(response_time, 2),
                'tables_found': len(tables),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.results['database'] = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_redis(self):
        """Check Redis connectivity and performance"""
        try:
            start_time = time.time()
            
            r = redis.Redis(host="localhost", port=6379, db=0)
            
            # Test ping
            await r.ping()
            
            # Test set/get
            test_key = f"health_check_{int(time.time())}"
            await r.set(test_key, "test_value", ex=60)
            value = await r.get(test_key)
            await r.delete(test_key)
            
            await r.close()
            
            response_time = (time.time() - start_time) * 1000
            
            self.results['redis'] = {
                'status': 'healthy',
                'response_time_ms': round(response_time, 2),
                'test_successful': value.decode() == "test_value",
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.results['redis'] = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_encryption(self):
        """Check encryption/decryption functionality"""
        try:
            from capabilities.composition.central_configuration.security_engine import SecurityEngine
            
            security = SecurityEngine(tenant_id="health_check")
            
            # Test encryption/decryption
            test_value = "health_check_test_value"
            encrypted = await security.encrypt_config_value("test_key", test_value, "health_check")
            decrypted = await security.decrypt_config_value(encrypted, "health_check")
            
            self.results['encryption'] = {
                'status': 'healthy',
                'test_successful': decrypted == test_value,
                'quantum_crypto_available': hasattr(security, 'quantum_crypto_available'),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.results['encryption'] = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def check_ml_models(self):
        """Check ML model availability and performance"""
        try:
            from capabilities.composition.central_configuration.ml_models_advanced import MLEngine
            
            ml_engine = MLEngine(tenant_id="health_check")
            
            # Check if models are loaded
            models = await ml_engine.list_models()
            
            # Test prediction if models available
            prediction_test = False
            if models:
                try:
                    # Test with dummy features
                    test_features = {
                        'config_change_frequency': 0.5,
                        'user_activity_score': 0.3,
                        'time_of_day_normalized': 0.6
                    }
                    
                    result = await ml_engine.predict_anomaly(
                        models[0]['name'], 
                        test_features
                    )
                    prediction_test = True
                except:
                    pass
            
            self.results['ml_models'] = {
                'status': 'healthy' if models else 'degraded',
                'models_loaded': len(models),
                'prediction_test': prediction_test,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.results['ml_models'] = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def run_all_checks(self):
        """Run all health checks"""
        await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_encryption(),
            self.check_ml_models(),
            return_exceptions=True
        )
        
        # Overall status
        statuses = [check['status'] for check in self.results.values()]
        
        if all(status == 'healthy' for status in statuses):
            overall_status = 'healthy'
        elif any(status == 'unhealthy' for status in statuses):
            overall_status = 'unhealthy'
        else:
            overall_status = 'degraded'
        
        self.results['overall'] = {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return self.results

# Usage
async def main():
    checker = HealthChecker()
    results = await checker.run_all_checks()
    
    print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    if results['overall']['status'] == 'healthy':
        exit(0)
    elif results['overall']['status'] == 'degraded':
        exit(1)
    else:
        exit(2)

if __name__ == "__main__":
    asyncio.run(main())
```

### Log Analysis Script

```bash
#!/bin/bash
# APG Central Configuration Log Analysis Script

LOG_DIR="/var/log/apg-central-config"
ANALYSIS_PERIOD="1h"  # Last 1 hour

echo "=== APG Central Configuration Log Analysis ==="
echo "Analysis Period: Last $ANALYSIS_PERIOD"
echo "Timestamp: $(date)"
echo

# Error Analysis
echo "=== Error Analysis ==="
echo "Top 10 Errors:"
grep -r "ERROR" $LOG_DIR --include="*.log" | \
  awk -F':' '{print $NF}' | \
  sort | uniq -c | sort -nr | head -10

echo
echo "Database Errors:"
grep -r "database\|postgresql\|asyncpg" $LOG_DIR --include="*.log" | \
  grep -i error | wc -l

echo "Redis Errors:"
grep -r "redis" $LOG_DIR --include="*.log" | \
  grep -i error | wc -l

echo "Encryption Errors:"
grep -r "encryption\|decrypt\|crypto" $LOG_DIR --include="*.log" | \
  grep -i error | wc -l

# Performance Analysis
echo
echo "=== Performance Analysis ==="
echo "Slow Queries (>1000ms):"
grep -r "took.*[0-9]\{4,\}ms" $LOG_DIR --include="*.log" | wc -l

echo "Memory Warnings:"
grep -r "memory.*warning\|high memory usage" $LOG_DIR --include="*.log" | wc -l

echo "Rate Limiting:"
grep -r "rate limit\|too many requests" $LOG_DIR --include="*.log" | wc -l

# Security Analysis
echo
echo "=== Security Analysis ==="
echo "Authentication Failures:"
grep -r "authentication.*failed\|invalid.*token" $LOG_DIR --include="*.log" | wc -l

echo "Anomalies Detected:"
grep -r "anomaly.*detected\|anomalous" $LOG_DIR --include="*.log" | wc -l

echo "Failed Logins:"
grep -r "login.*failed\|unauthorized" $LOG_DIR --include="*.log" | wc -l

# Configuration Changes
echo
echo "=== Configuration Activity ==="
echo "Configuration Changes:"
grep -r "config.*changed\|configuration.*updated" $LOG_DIR --include="*.log" | wc -l

echo "Rollbacks:"
grep -r "rollback\|reverted" $LOG_DIR --include="*.log" | wc -l

echo "Conflicts:"
grep -r "conflict.*detected\|conflict.*resolved" $LOG_DIR --include="*.log" | wc -l

# System Health
echo
echo "=== System Health ==="
echo "Service Starts:"
grep -r "service.*started\|server.*started" $LOG_DIR --include="*.log" | wc -l

echo "Service Stops:"
grep -r "service.*stopped\|server.*stopped" $LOG_DIR --include="*.log" | wc -l

echo "Health Check Failures:"
grep -r "health.*check.*failed\|unhealthy" $LOG_DIR --include="*.log" | wc -l
```

## Getting Help

### Support Channels

1. **Documentation**: Check the comprehensive docs in `/docs/`
2. **Health Checks**: Run the health check script regularly
3. **Log Analysis**: Use the log analysis script for diagnostics
4. **Community**: Join the APG community forums
5. **Enterprise Support**: Contact nyimbi@gmail.com for enterprise support

### Reporting Issues

When reporting issues, please include:

1. **Error Messages**: Full error messages and stack traces
2. **Environment Info**: OS, Python version, dependency versions
3. **Configuration**: Relevant configuration settings (sanitized)
4. **Steps to Reproduce**: Detailed steps to reproduce the issue
5. **Expected vs Actual**: What you expected vs what happened
6. **Logs**: Relevant log entries around the time of the issue

### Emergency Procedures

#### Service Recovery

```bash
#!/bin/bash
# Emergency service recovery script

echo "=== APG Central Configuration Emergency Recovery ==="

# Stop services
echo "Stopping services..."
sudo systemctl stop apg-central-config
sudo systemctl stop postgresql
sudo systemctl stop redis-server

# Check and repair database
echo "Checking database integrity..."
sudo -u postgres pg_dump central_config > /tmp/config_backup.sql
sudo -u postgres psql -d central_config -c "REINDEX DATABASE central_config;"

# Clear Redis cache
echo "Clearing Redis cache..."
redis-cli FLUSHDB

# Restart services
echo "Restarting services..."
sudo systemctl start postgresql
sudo systemctl start redis-server
sudo systemctl start apg-central-config

# Verify services
echo "Verifying services..."
sleep 10
sudo systemctl status apg-central-config
python3 health_check.py

echo "Recovery complete. Check logs for any remaining issues."
```

#### Data Recovery

```bash
#!/bin/bash
# Data recovery from backups

BACKUP_DIR="/var/backups/apg-central-config"
RECOVERY_DATE="2025-01-01"  # Adjust as needed

echo "=== Data Recovery Procedure ==="
echo "Recovery Date: $RECOVERY_DATE"

# Stop service
sudo systemctl stop apg-central-config

# Restore database
echo "Restoring database..."
sudo -u postgres dropdb central_config
sudo -u postgres createdb central_config
sudo -u postgres psql central_config < "$BACKUP_DIR/db_backup_$RECOVERY_DATE.sql"

# Restore configuration files
echo "Restoring configuration files..."
cp "$BACKUP_DIR/config_backup_$RECOVERY_DATE.tar.gz" /tmp/
cd /etc/apg && sudo tar -xzf "/tmp/config_backup_$RECOVERY_DATE.tar.gz"

# Restore encryption keys
echo "Restoring encryption keys..."
sudo cp "$BACKUP_DIR/keys_backup_$RECOVERY_DATE/"* /etc/apg/keys/
sudo chown -R apg:apg /etc/apg/keys/
sudo chmod 600 /etc/apg/keys/*

# Restart service
sudo systemctl start apg-central-config

echo "Data recovery complete. Verify system functionality."
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create comprehensive API documentation for Central Configuration", "status": "completed", "priority": "high", "id": "api_docs"}, {"content": "Write deployment and installation guide", "status": "completed", "priority": "high", "id": "deployment_guide"}, {"content": "Create user guide and configuration examples", "status": "completed", "priority": "high", "id": "user_guide"}, {"content": "Document enterprise integrations and connectors", "status": "completed", "priority": "medium", "id": "enterprise_docs"}, {"content": "Create security and encryption documentation", "status": "completed", "priority": "medium", "id": "security_docs"}, {"content": "Write real-time synchronization guide", "status": "completed", "priority": "medium", "id": "realtime_docs"}, {"content": "Document ML models and AutoML capabilities", "status": "completed", "priority": "medium", "id": "ml_docs"}, {"content": "Create troubleshooting and FAQ documentation", "status": "completed", "priority": "low", "id": "troubleshooting_docs"}]