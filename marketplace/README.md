# APG Capability Marketplace

The APG Capability Marketplace is a community-driven platform for discovering, sharing, and managing reusable capabilities for the APG (Application Programming Generation) system. It enables developers to contribute capabilities and discover solutions built by the community.

## Overview

The marketplace consists of several key components:

- **Core Marketplace System** (`capability_marketplace.py`) - The main marketplace engine
- **Web API** (`web_api.py`) - RESTful API for web integration
- **CLI Interface** (`cli.py`) - Command-line tools for marketplace interaction
- **Security Validation** - Built-in security and quality checks
- **Discovery Engine** - Intelligent search and recommendation system

## Quick Start

### Installation

```bash
# Install dependencies
pip install fastapi uvicorn click rich pydantic uuid-extensions

# Optional: For enhanced functionality
pip install pandas numpy requests
```

### Using the CLI

```bash
# List available capabilities
python marketplace/cli.py list

# Search for capabilities
python marketplace/cli.py search "web authentication"

# Show detailed information
python marketplace/cli.py show <capability_id>

# Download a capability
python marketplace/cli.py download <capability_id>

# Submit a new capability
python marketplace/cli.py submit my_capability.json
```

### Using the Web API

```bash
# Start the API server
python marketplace/web_api.py

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Using the Python API

```python
import asyncio
from marketplace.capability_marketplace import CapabilityMarketplace

async def main():
    # Initialize marketplace
    marketplace = CapabilityMarketplace("./marketplace_data")
    
    # Search for capabilities
    capabilities = await marketplace.search_capabilities("machine learning")
    
    # Get recommendations
    recommendations = await marketplace.get_recommendations(limit=5)
    
    print(f"Found {len(capabilities)} capabilities")

asyncio.run(main())
```

## Capability Development Guide

### 1. Capability Structure

A capability is a self-contained, reusable component that provides specific functionality. Each capability includes:

- **Core Code** - The main implementation
- **Documentation** - Usage instructions and API reference
- **Examples** - Sample usage and integration patterns
- **Tests** - Validation and quality assurance
- **Dependencies** - Required packages and versions
- **Metadata** - Author, license, categories, and tags

### 2. Creating a Capability

#### Step 1: Define Your Capability

Create a JSON file describing your capability:

```json
{
  "name": "web_authentication",
  "display_name": "Web Authentication Helper",
  "description": "Secure web authentication using JWT tokens and OAuth2 integration for modern web applications with session management and security best practices.",
  "detailed_description": "This capability provides comprehensive web authentication including JWT token generation and verification, OAuth2 flows, secure password hashing, session management, and implements security best practices for web applications.",
  "category": "web_development",
  "tags": ["authentication", "jwt", "oauth2", "security", "web"],
  "keywords": ["auth", "login", "token", "security", "web", "api"],
  "author": "Your Name",
  "author_email": "your.email@example.com",
  "organization": "Your Organization",
  "license": "mit",
  "homepage": "https://github.com/yourusername/web-auth",
  "repository": "https://github.com/yourusername/web-auth.git",
  "capability_code": "... your Python code here ...",
  "example_usage": "... usage examples ...",
  "documentation": "... detailed documentation ...",
  "dependencies": [
    {
      "name": "PyJWT",
      "version_constraint": ">=2.0.0",
      "optional": false,
      "description": "JWT token handling"
    }
  ],
  "platforms": ["linux", "windows", "macos"]
}
```

#### Step 2: Implement Your Code

Your capability code should be well-structured and follow these guidelines:

```python
"""
Web Authentication Capability
============================

Provides secure authentication for web applications.
"""

import jwt
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class WebAuthenticator:
    """
    Web authentication helper with JWT and OAuth2 support.
    
    This class provides secure authentication mechanisms including:
    - JWT token generation and verification
    - Password hashing and verification
    - Session management utilities
    """
    
    def __init__(self, secret_key: str):
        """
        Initialize the authenticator.
        
        Args:
            secret_key: Secret key for JWT signing
        """
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str, expires_hours: int = 24) -> str:
        """
        Generate a JWT token for a user.
        
        Args:
            user_id: Unique user identifier
            expires_hours: Token expiration time in hours
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid, None if invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password securely.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        return self.hash_password(password) == hashed

# Capability metadata for APG integration
CAPABILITY_INFO = {
    "name": "web_authentication",
    "version": "1.0.0",
    "provides": ["WebAuthenticator"],
    "integrates_with": ["flask", "fastapi", "django"],
    "apg_templates": ["web_app", "api_service", "microservice"]
}
```

#### Step 3: Add Documentation

Include comprehensive documentation in Markdown format:

```markdown
# Web Authentication Capability

## Overview

This capability provides secure authentication mechanisms for web applications, including JWT token management, OAuth2 integration, and password security.

## Features

- **JWT Token Management**: Generate and verify JSON Web Tokens
- **Password Security**: Secure password hashing and verification
- **OAuth2 Integration**: Ready for OAuth2 flow implementation
- **Session Management**: Utilities for managing user sessions
- **Security Best Practices**: Implements current security standards

## Quick Start

```python
from web_authentication import WebAuthenticator

# Initialize authenticator
auth = WebAuthenticator("your-secret-key-here")

# Generate token for user
token = auth.generate_token("user123")

# Verify token
payload = auth.verify_token(token)
if payload:
    print(f"User ID: {payload['user_id']}")

# Password handling
hashed_password = auth.hash_password("user_password")
is_valid = auth.verify_password("user_password", hashed_password)
```

## API Reference

### WebAuthenticator

#### `__init__(secret_key: str)`
Initialize the authenticator with a secret key.

#### `generate_token(user_id: str, expires_hours: int = 24) -> str`
Generate a JWT token for a user.

#### `verify_token(token: str) -> Optional[Dict[str, Any]]`
Verify and decode a JWT token.

#### `hash_password(password: str) -> str`
Hash a password securely using SHA-256.

#### `verify_password(password: str, hashed: str) -> bool`
Verify a password against its hash.

## Integration with APG Templates

This capability integrates seamlessly with APG web application templates:

### Flask Integration
```python
from flask import Flask, request, jsonify
from web_authentication import WebAuthenticator

app = Flask(__name__)
auth = WebAuthenticator(app.config['SECRET_KEY'])

@app.route('/login', methods=['POST'])
def login():
    # Authentication logic here
    token = auth.generate_token(user_id)
    return jsonify({'token': token})
```

### FastAPI Integration
```python
from fastapi import FastAPI, Depends, HTTPException
from web_authentication import WebAuthenticator

app = FastAPI()
auth = WebAuthenticator("your-secret-key")

def get_current_user(token: str = Depends(get_token)):
    payload = auth.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload['user_id']
```

## Security Considerations

- Always use strong, unique secret keys
- Implement proper token expiration
- Use HTTPS in production
- Consider implementing refresh tokens for long-lived sessions
- Follow OWASP security guidelines

## Dependencies

- `PyJWT>=2.0.0`: JWT token handling
- `Python>=3.8`: Required Python version

## License

MIT License - see LICENSE file for details.
```

#### Step 4: Add Example Usage

Provide clear, practical examples:

```python
"""
Web Authentication Capability - Examples
========================================

This file demonstrates various usage patterns for the web authentication capability.
"""

from web_authentication import WebAuthenticator
import json

# Example 1: Basic Authentication
def basic_auth_example():
    """Basic authentication workflow"""
    
    # Initialize authenticator
    auth = WebAuthenticator("my-secret-key-12345")
    
    # User registration simulation
    user_id = "user_12345"
    password = "secure_password_123"
    
    # Hash password for storage
    hashed_password = auth.hash_password(password)
    print(f"Hashed password: {hashed_password}")
    
    # User login simulation
    login_password = "secure_password_123"
    if auth.verify_password(login_password, hashed_password):
        # Generate token on successful login
        token = auth.generate_token(user_id, expires_hours=24)
        print(f"Login successful. Token: {token}")
        
        # Verify token (as middleware would do)
        payload = auth.verify_token(token)
        if payload:
            print(f"Token valid. User ID: {payload['user_id']}")
        else:
            print("Token invalid or expired")
    else:
        print("Login failed - invalid password")

# Example 2: Flask Integration
def flask_integration_example():
    """Example of integrating with Flask"""
    
    flask_code = '''
from flask import Flask, request, jsonify, g
from functools import wraps
from web_authentication import WebAuthenticator

app = Flask(__name__)
auth = WebAuthenticator(app.config.get('SECRET_KEY', 'dev-key'))

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token:
            token = token.replace('Bearer ', '')
            payload = auth.verify_token(token)
            if payload:
                g.current_user = payload['user_id']
                return f(*args, **kwargs)
        return jsonify({'message': 'Token required'}), 401
    return decorated

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Verify credentials (pseudo-code)
    if verify_user_credentials(username, password):
        token = auth.generate_token(username)
        return jsonify({'token': token})
    else:
        return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/protected')
@token_required
def protected():
    return jsonify({'message': f'Hello {g.current_user}'})
    '''
    
    print("Flask Integration Example:")
    print(flask_code)

# Example 3: Token Management
def token_management_example():
    """Advanced token management"""
    
    auth = WebAuthenticator("advanced-secret-key")
    
    # Short-lived access token
    access_token = auth.generate_token("user123", expires_hours=1)
    
    # Longer-lived refresh token (would need separate implementation)
    refresh_token = auth.generate_token(f"refresh_user123", expires_hours=24*7)
    
    print(f"Access token (1h): {access_token}")
    print(f"Refresh token (7d): {refresh_token}")
    
    # Token verification with error handling
    def verify_and_handle_token(token):
        payload = auth.verify_token(token)
        if payload:
            return {"valid": True, "user_id": payload["user_id"]}
        else:
            return {"valid": False, "error": "Token expired or invalid"}
    
    # Test verification
    result = verify_and_handle_token(access_token)
    print(f"Token verification result: {result}")

if __name__ == "__main__":
    print("Web Authentication Capability Examples")
    print("=" * 40)
    
    print("\n1. Basic Authentication:")
    basic_auth_example()
    
    print("\n2. Flask Integration:")
    flask_integration_example()
    
    print("\n3. Token Management:")
    token_management_example()
```

#### Step 5: Write Tests

Include comprehensive test cases:

```python
"""
Web Authentication Capability - Tests
====================================

Comprehensive test suite for the web authentication capability.
"""

import unittest
from datetime import datetime, timedelta
import jwt
from web_authentication import WebAuthenticator

class TestWebAuthenticator(unittest.TestCase):
    """Test cases for WebAuthenticator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.secret_key = "test-secret-key-12345"
        self.auth = WebAuthenticator(self.secret_key)
        self.test_user_id = "test_user_123"
    
    def test_token_generation(self):
        """Test JWT token generation"""
        token = self.auth.generate_token(self.test_user_id)
        
        # Token should be a non-empty string
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 0)
        
        # Token should be valid JWT
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            self.assertEqual(payload['user_id'], self.test_user_id)
        except jwt.InvalidTokenError:
            self.fail("Generated token is not valid JWT")
    
    def test_token_verification(self):
        """Test JWT token verification"""
        # Generate token
        token = self.auth.generate_token(self.test_user_id)
        
        # Verify token
        payload = self.auth.verify_token(token)
        
        self.assertIsNotNone(payload)
        self.assertEqual(payload['user_id'], self.test_user_id)
        self.assertIn('exp', payload)
        self.assertIn('iat', payload)
    
    def test_invalid_token_verification(self):
        """Test verification of invalid tokens"""
        # Test completely invalid token
        invalid_payload = self.auth.verify_token("invalid.token.here")
        self.assertIsNone(invalid_payload)
        
        # Test token with wrong secret
        wrong_auth = WebAuthenticator("wrong-secret")
        valid_token = self.auth.generate_token(self.test_user_id)
        wrong_payload = wrong_auth.verify_token(valid_token)
        self.assertIsNone(wrong_payload)
    
    def test_token_expiration(self):
        """Test token expiration"""
        # Generate short-lived token (for testing, we'll manipulate it)
        token = self.auth.generate_token(self.test_user_id, expires_hours=24)
        
        # Manually create an expired token
        expired_payload = {
            'user_id': self.test_user_id,
            'exp': datetime.utcnow() - timedelta(hours=1),  # Expired 1 hour ago
            'iat': datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(expired_payload, self.secret_key, algorithm='HS256')
        
        # Verify expired token returns None
        result = self.auth.verify_token(expired_token)
        self.assertIsNone(result)
    
    def test_password_hashing(self):
        """Test password hashing"""
        password = "test_password_123"
        hashed = self.auth.hash_password(password)
        
        # Hash should be a string
        self.assertIsInstance(hashed, str)
        
        # Hash should be deterministic
        hashed2 = self.auth.hash_password(password)
        self.assertEqual(hashed, hashed2)
        
        # Different passwords should produce different hashes
        different_hash = self.auth.hash_password("different_password")
        self.assertNotEqual(hashed, different_hash)
    
    def test_password_verification(self):
        """Test password verification"""
        password = "test_password_123"
        hashed = self.auth.hash_password(password)
        
        # Correct password should verify
        self.assertTrue(self.auth.verify_password(password, hashed))
        
        # Wrong password should not verify
        self.assertFalse(self.auth.verify_password("wrong_password", hashed))
        
        # Empty password should not verify
        self.assertFalse(self.auth.verify_password("", hashed))
    
    def test_custom_expiration(self):
        """Test custom token expiration times"""
        # Test 1-hour expiration
        token_1h = self.auth.generate_token(self.test_user_id, expires_hours=1)
        payload_1h = self.auth.verify_token(token_1h)
        
        # Check expiration is approximately 1 hour from now
        exp_time = datetime.fromtimestamp(payload_1h['exp'])
        expected_exp = datetime.utcnow() + timedelta(hours=1)
        time_diff = abs((exp_time - expected_exp).total_seconds())
        self.assertLess(time_diff, 60)  # Within 1 minute tolerance
    
    def test_multiple_users(self):
        """Test handling multiple users"""
        users = ["user1", "user2", "user3"]
        tokens = {}
        
        # Generate tokens for multiple users
        for user in users:
            tokens[user] = self.auth.generate_token(user)
        
        # Verify each token returns correct user
        for user, token in tokens.items():
            payload = self.auth.verify_token(token)
            self.assertIsNotNone(payload)
            self.assertEqual(payload['user_id'], user)

if __name__ == '__main__':
    unittest.main()
```

### 3. Capability Categories

Choose the appropriate category for your capability:

- **web_development** - Web frameworks, APIs, authentication
- **ai_ml** - Machine learning, AI models, data science
- **iot_hardware** - IoT devices, sensors, hardware integration
- **business_intelligence** - Analytics, reporting, dashboards
- **cloud_integration** - Cloud services, deployment, scaling
- **security_compliance** - Security tools, compliance, encryption
- **performance_monitoring** - Metrics, logging, optimization
- **devops_deployment** - CI/CD, containerization, infrastructure
- **data_processing** - ETL, data transformation, pipelines
- **mobile_development** - Mobile apps, responsive design
- **blockchain** - Cryptocurrency, smart contracts, DeFi
- **gaming** - Game development, graphics, physics
- **healthcare** - Medical applications, HIPAA compliance
- **finance** - Financial calculations, trading, compliance
- **education** - Learning management, course delivery
- **custom** - Other specialized capabilities

### 4. Quality Guidelines

To ensure high-quality capabilities:

#### Code Quality
- Use clear, descriptive variable and function names
- Include comprehensive docstrings
- Follow Python PEP 8 style guidelines
- Include type hints where appropriate
- Handle errors gracefully
- Write modular, reusable code

#### Security
- Avoid dangerous functions like `eval()` and `exec()`
- Don't hardcode secrets or credentials
- Validate all inputs
- Use secure coding practices
- Include security considerations in documentation

#### Documentation
- Provide clear overview and purpose
- Include comprehensive API documentation
- Add practical usage examples
- Document integration patterns
- List all dependencies
- Include troubleshooting guide

#### Testing
- Write comprehensive test cases
- Test both success and failure scenarios
- Include edge cases
- Test integration with other systems
- Validate performance characteristics

### 5. Submission Process

#### Via CLI
```bash
python marketplace/cli.py submit my_capability.json
```

#### Via API
```bash
curl -X POST http://localhost:8000/capabilities \
  -H "Content-Type: application/json" \
  -d @my_capability.json
```

#### Via Python API
```python
import asyncio
from marketplace.capability_marketplace import CapabilityMarketplace, MarketplaceCapability

async def submit_capability():
    marketplace = CapabilityMarketplace()
    
    # Create capability object
    capability = MarketplaceCapability(
        name="my_capability",
        description="My awesome capability...",
        # ... other fields
    )
    
    # Submit
    result = await marketplace.submit_capability(capability)
    if result['success']:
        print(f"Capability submitted: {result['capability_id']}")
    else:
        print(f"Submission failed: {result['errors']}")

asyncio.run(submit_capability())
```

## Discovery and Search

### 1. Basic Search

Search capabilities by keywords, descriptions, and metadata:

```bash
# CLI search
python marketplace/cli.py search "machine learning"
python marketplace/cli.py search "web auth" --category web_development

# API search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "max_results": 10}'
```

### 2. Category Browsing

Browse capabilities by category:

```bash
# List all categories
python marketplace/cli.py categories

# Filter by category
python marketplace/cli.py list --category ai_ml
```

### 3. Recommendations

Get personalized recommendations:

```bash
# Based on a capability you're viewing
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"based_on_capability": "capability_id_here", "limit": 5}'

# Based on your download history
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_history": ["cap1", "cap2"], "limit": 5}'
```

### 4. Advanced Filtering

Use advanced filters for precise discovery:

- **Rating filter**: Only show highly-rated capabilities
- **Platform filter**: Filter by supported platforms
- **License filter**: Filter by license type
- **Author filter**: Find capabilities by specific authors
- **Tag filter**: Use tags for fine-grained filtering

## API Reference

### REST API Endpoints

#### Capabilities
- `GET /capabilities` - List capabilities
- `POST /capabilities` - Submit new capability
- `GET /capabilities/{id}` - Get capability details
- `PUT /capabilities/{id}` - Update capability
- `POST /capabilities/{id}/publish` - Publish capability
- `POST /capabilities/{id}/download` - Download capability

#### Search and Discovery
- `POST /search` - Search capabilities
- `POST /recommendations` - Get recommendations
- `GET /categories` - List categories
- `GET /licenses` - List license types

#### Ratings and Reviews
- `POST /capabilities/{id}/ratings` - Add rating
- `GET /capabilities/{id}/ratings` - Get ratings

#### Analytics
- `GET /stats` - Marketplace statistics
- `GET /health` - Health check

### CLI Commands

#### Listing and Discovery
```bash
marketplace list [--category CATEGORY] [--status STATUS] [--author AUTHOR]
marketplace search QUERY [--category CATEGORY] [--max-results N]
marketplace show CAPABILITY_ID [--include-code]
```

#### Capability Management
```bash
marketplace download CAPABILITY_ID [--output-dir DIR]
marketplace submit CAPABILITY_FILE
marketplace publish CAPABILITY_ID
marketplace rate CAPABILITY_ID RATING [--review TEXT]
```

#### Information
```bash
marketplace stats
marketplace categories
marketplace licenses
```

## Best Practices

### For Capability Authors

1. **Start Simple**: Begin with a focused, well-defined capability
2. **Documentation First**: Write clear documentation before coding
3. **Security Mindset**: Always consider security implications
4. **Test Thoroughly**: Include comprehensive tests
5. **Version Wisely**: Use semantic versioning for updates
6. **Community Focus**: Design for reusability and community benefit

### For Capability Users

1. **Read Documentation**: Always review capability documentation
2. **Check Ratings**: Look at community ratings and reviews
3. **Verify Security**: Review code for security implications
4. **Test Integration**: Test capabilities in your environment
5. **Provide Feedback**: Rate and review capabilities you use
6. **Contribute Back**: Share improvements with the community

### For Marketplace Administrators

1. **Monitor Quality**: Regularly review submission quality
2. **Security Scanning**: Implement automated security checks
3. **Community Building**: Foster a collaborative environment
4. **Performance Monitoring**: Ensure marketplace performance
5. **Data Backup**: Maintain regular backups of marketplace data

## Troubleshooting

### Common Issues

#### Capability Submission Fails
- Check required fields are complete
- Verify description meets minimum length
- Ensure code passes security validation
- Review dependency specifications

#### Search Returns No Results
- Try broader search terms
- Check category filters
- Verify capability status (published vs draft)
- Use tag-based search

#### Download Issues
- Ensure capability is published
- Check network connectivity
- Verify user permissions
- Try downloading to different directory

#### API Connection Problems
- Verify API server is running
- Check network connectivity
- Confirm API endpoint URLs
- Review authentication if required

### Getting Help

1. **Documentation**: Check this guide and API docs
2. **Community Forum**: Ask questions in community channels
3. **Issue Tracker**: Report bugs and feature requests
4. **Contact Support**: Reach out to maintainers

## Contributing

We welcome contributions to the APG Capability Marketplace!

### Ways to Contribute

1. **Submit Capabilities**: Share useful capabilities with the community
2. **Improve Documentation**: Help make documentation clearer
3. **Report Issues**: Help identify and fix problems
4. **Feature Requests**: Suggest new marketplace features
5. **Code Contributions**: Contribute to marketplace codebase

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/apg-marketplace.git
cd apg-marketplace

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Start development server
python marketplace/web_api.py
```

## License

The APG Capability Marketplace is released under the MIT License. Individual capabilities may have their own licenses - always check the license information for each capability before use.

## Changelog

### Version 1.0.0
- Initial marketplace release
- Core capability management
- Web API and CLI interfaces
- Security validation system
- Discovery and recommendation engine
- Community rating system

---

*For more information, visit the [APG Project Documentation](../docs/) or contact the development team.*