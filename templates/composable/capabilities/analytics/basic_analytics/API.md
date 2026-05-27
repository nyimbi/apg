# Basic Analytics API Documentation

## Endpoints

### Health Check

```
GET /analytics/basic_analytics/health
```

Returns the health status of the Basic Analytics capability.

**Response:**
```json
{
  "status": "healthy",
  "capability": "Basic Analytics",
  "version": "1.0.0"
}
```

## Authentication

This capability supports the following authentication methods:

- Basic Authentication
- JWT Tokens
- API Keys

## Error Handling

All API endpoints return standard HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Rate Limiting

API endpoints are rate limited to prevent abuse:

- 100 requests per minute per IP address
- 1000 requests per hour per authenticated user

## Examples

### cURL Examples

```bash
# Health check
curl -X GET "${APG_RUNTIME_URL}/analytics/basic_analytics/health"

curl -X GET "${APG_RUNTIME_URL}/analytics/basic_analytics/status"
```

### Python Examples

```python
import os
import requests

base_url = os.environ.get('APG_RUNTIME_URL', '').rstrip('/')

# Health check
response = requests.get(f'{base_url}/analytics/basic_analytics/health')
print(response.json())

status = requests.get(f'{base_url}/analytics/basic_analytics/status')
status.raise_for_status()
print(status.json())
```
