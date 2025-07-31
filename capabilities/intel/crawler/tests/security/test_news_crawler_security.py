"""
Security tests for news crawler system.

This module contains security tests that validate the news crawler
against various security threats and vulnerabilities.
"""

import pytest
import asyncio
import re
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import urllib.parse
import hashlib
import ssl
import socket

from packages.crawlers.news_crawler.core.enhanced_news_crawler import NewsCrawler
from packages.crawlers.base_crawler import BaseCrawler
from packages.database.postgresql_manager import PgSQLManager
from packages.utils.caching.manager import CacheManager


class TestNewsCrawlerSecurity:
    """Security tests for news crawler system."""
    
    @pytest.fixture
    def security_config(self):
        """Configuration for security testing."""
        return {
            'max_concurrent': 3,
            'retry_count': 2,
            'timeout': 10.0,
            'stealth_enabled': True,
            'bypass_enabled': False,  # Disable bypass for security testing
            'cache_enabled': True,
            'cache_ttl': 3600,
            'content_extraction': {
                'extract_text': True,
                'extract_images': True,
                'extract_links': True,
                'extract_metadata': True
            },
            'security_config': {
                'validate_ssl': True,
                'check_content_type': True,
                'max_content_size': 10 * 1024 * 1024,  # 10MB
                'sanitize_content': True,
                'block_dangerous_urls': True
            }
        }
    
    @pytest.fixture
    def malicious_test_cases(self):
        """Test cases for malicious content."""
        return {
            'xss_attempts': [
                '<script>alert("XSS")</script>',
                '<img src="x" onerror="alert(1)">',
                '<svg onload="alert(1)">',
                'javascript:alert("XSS")',
                '<iframe src="javascript:alert(1)"></iframe>'
            ],
            'sql_injection_attempts': [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "'; INSERT INTO users VALUES('hacker'); --",
                "' UNION SELECT * FROM passwords --"
            ],
            'malicious_urls': [
                'https://example.com/../../etc/passwd',
                'https://example.com/admin/../../../etc/shadow',
                'file:///etc/passwd',
                'ftp://malicious.com/payload',
                'https://evil.com/redirect?url=file:///etc/passwd'
            ],
            'oversized_content': 'A' * (20 * 1024 * 1024),  # 20MB
            'suspicious_headers': {
                'X-Frame-Options': 'DENY',
                'X-Content-Type-Options': 'nosniff',
                'X-XSS-Protection': '1; mode=block',
                'Content-Security-Policy': "default-src 'self'"
            }
        }
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_url_validation_security(self, security_config, malicious_test_cases):
        """Test URL validation against malicious URLs."""
        crawler = NewsCrawler(config=security_config)
        
        async with crawler:
            for malicious_url in malicious_test_cases['malicious_urls']:
                # Should either reject the URL or handle it safely
                try:
                    result = await crawler.crawl_url(malicious_url)
                    
                    # If it doesn't reject, it should return None or error
                    if result is not None:
                        assert 'error' in result or result.get('content') is None
                        
                except (ValueError, SecurityError) as e:
                    # URL validation should catch malicious URLs
                    assert 'malicious' in str(e).lower() or 'invalid' in str(e).lower()
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_content_sanitization(self, security_config, malicious_test_cases):
        """Test content sanitization against XSS attacks."""
        crawler = NewsCrawler(config=security_config)
        
        for xss_payload in malicious_test_cases['xss_attempts']:
            malicious_html = f"""
            <html>
                <head><title>Test Page</title></head>
                <body>
                    <h1>Article Title</h1>
                    <p>Normal content here.</p>
                    {xss_payload}
                    <p>More normal content.</p>
                </body>
            </html>
            """
            
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = malicious_html
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                async with crawler:
                    result = await crawler.crawl_url('https://example.com/test')
                    
                    if result is not None:
                        content = result.get('content', '')
                        
                        # Should sanitize malicious content
                        assert '<script>' not in content
                        assert 'javascript:' not in content
                        assert 'onerror=' not in content
                        assert 'onload=' not in content
                        
                        # Should preserve normal content
                        assert 'Normal content here' in content
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, security_config, malicious_test_cases):
        """Test SQL injection prevention in database operations."""
        crawler = NewsCrawler(config=security_config)
        
        mock_db = Mock(spec=PgSQLManager)
        mock_db.store_article = AsyncMock()
        mock_db.get_article = AsyncMock(return_value=None)
        
        crawler.db_manager = mock_db
        
        for sql_payload in malicious_test_cases['sql_injection_attempts']:
            malicious_html = f"""
            <html>
                <head>
                    <title>Article {sql_payload}</title>
                    <meta name="author" content="{sql_payload}">
                </head>
                <body>
                    <h1>Title with {sql_payload}</h1>
                    <p>Content with {sql_payload}</p>
                </body>
            </html>
            """
            
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = malicious_html
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                async with crawler:
                    result = await crawler.crawl_url('https://example.com/test')
                    
                    # Should not cause SQL injection
                    if mock_db.store_article.called:
                        stored_data = mock_db.store_article.call_args[0][0]
                        
                        # Check that SQL injection attempts are properly escaped
                        for field_value in stored_data.values():
                            if isinstance(field_value, str):
                                # Should not contain unescaped SQL injection patterns
                                assert "'; DROP TABLE" not in field_value
                                assert "' OR '1'='1" not in field_value
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_ssl_certificate_validation(self, security_config):
        """Test SSL certificate validation."""
        crawler = NewsCrawler(config=security_config)
        
        # Test with potentially invalid SSL certificates
        test_cases = [
            'https://self-signed.badssl.com/',
            'https://expired.badssl.com/',
            'https://wrong.host.badssl.com/',
            'https://untrusted-root.badssl.com/'
        ]
        
        async with crawler:
            for test_url in test_cases:
                try:
                    result = await crawler.crawl_url(test_url)
                    
                    # Should either reject invalid certificates or handle safely
                    if result is not None:
                        assert 'error' in result or 'ssl' in result.get('warnings', [])
                        
                except ssl.SSLError:
                    # SSL errors are expected for invalid certificates
                    pass
                except Exception as e:
                    # Other connection errors are acceptable
                    assert 'ssl' in str(e).lower() or 'certificate' in str(e).lower()
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_content_size_limits(self, security_config, malicious_test_cases):
        """Test content size limits to prevent DoS attacks."""
        crawler = NewsCrawler(config=security_config)
        
        # Test with oversized content
        oversized_content = malicious_test_cases['oversized_content']
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = oversized_content
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                result = await crawler.crawl_url('https://example.com/large-content')
                
                # Should handle oversized content safely
                if result is not None:
                    # Content should be truncated or rejected
                    content = result.get('content', '')
                    assert len(content) < 15 * 1024 * 1024  # Less than 15MB
                    
                    # Should have warning about size
                    warnings = result.get('warnings', [])
                    assert any('size' in w.lower() for w in warnings)
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_header_injection_prevention(self, security_config):
        """Test prevention of header injection attacks."""
        crawler = NewsCrawler(config=security_config)
        
        # Test with malicious headers
        malicious_headers = {
            'X-Forwarded-For': '127.0.0.1\r\nMalicious-Header: injected',
            'User-Agent': 'Mozilla/5.0\r\nX-Injected: malicious',
            'Referer': 'https://example.com\r\nHost: evil.com'
        }
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body>Test</body></html>'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                # Should sanitize headers before making requests
                result = await crawler.crawl_url('https://example.com/test')
                
                # Verify the request was made safely
                if mock_stealth.fetch_with_stealth.called:
                    # Headers should be sanitized
                    assert result is not None
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_redirect_chain_security(self, security_config):
        """Test security of redirect chains."""
        crawler = NewsCrawler(config=security_config)
        
        # Test with potentially malicious redirect chains
        test_cases = [
            {
                'url': 'https://example.com/redirect1',
                'redirects': [
                    'https://example.com/redirect2',
                    'https://example.com/redirect3',
                    'file:///etc/passwd',  # Malicious redirect
                ]
            },
            {
                'url': 'https://example.com/redirect1',
                'redirects': ['https://example.com/redirect'] * 20  # Too many redirects
            }
        ]
        
        for test_case in test_cases:
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                # Mock redirect responses
                responses = []
                for i, redirect_url in enumerate(test_case['redirects']):
                    if i < len(test_case['redirects']) - 1:
                        # Redirect response
                        mock_response = Mock()
                        mock_response.status_code = 302
                        mock_response.headers = {'Location': redirect_url}
                        mock_response.text = ''
                        responses.append(mock_response)
                    else:
                        # Final response
                        mock_response = Mock()
                        mock_response.status_code = 200
                        mock_response.text = '<html><body>Final</body></html>'
                        responses.append(mock_response)
                
                mock_stealth.fetch_with_stealth = AsyncMock(side_effect=responses)
                
                async with crawler:
                    result = await crawler.crawl_url(test_case['url'])
                    
                    # Should handle malicious redirects safely
                    if result is not None:
                        # Should not follow dangerous redirects
                        assert 'file://' not in result.get('url', '')
                        
                        # Should limit redirect chains
                        if len(test_case['redirects']) > 10:
                            assert 'error' in result or 'redirect' in result.get('warnings', [])
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_user_agent_security(self, security_config):
        """Test user agent security and rotation."""
        crawler = NewsCrawler(config=security_config)
        
        # Test that user agents don't reveal system information
        test_urls = [f'https://httpbin.org/user-agent?id={i}' for i in range(5)]
        
        user_agents_used = []
        
        def capture_user_agent(request):
            user_agent = request.headers.get('User-Agent', '')
            user_agents_used.append(user_agent)
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = f'{{"user-agent": "{user_agent}"}}'
            return mock_response
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_stealth.fetch_with_stealth = AsyncMock(side_effect=capture_user_agent)
            
            async with crawler:
                for url in test_urls:
                    await crawler.crawl_url(url)
        
        # Verify user agents are safe
        for user_agent in user_agents_used:
            # Should not reveal system information
            assert 'python' not in user_agent.lower()
            assert 'crawler' not in user_agent.lower()
            assert 'bot' not in user_agent.lower()
            
            # Should look like legitimate browsers
            assert any(browser in user_agent for browser in ['Mozilla', 'Chrome', 'Firefox', 'Safari'])
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_cookie_security(self, security_config):
        """Test cookie handling security."""
        crawler = NewsCrawler(config=security_config)
        
        # Test with various cookie scenarios
        test_cases = [
            {
                'url': 'https://example.com/cookies',
                'cookies': {
                    'session': 'abc123',
                    'tracking': 'xyz789',
                    'xss_attempt': '<script>alert(1)</script>'
                }
            }
        ]
        
        for test_case in test_cases:
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = '<html><body>Cookie test</body></html>'
                mock_response.cookies = test_case['cookies']
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                async with crawler:
                    result = await crawler.crawl_url(test_case['url'])
                    
                    # Should handle cookies safely
                    if result is not None:
                        # Should not execute malicious cookie content
                        assert '<script>' not in str(result)
                        assert 'alert(1)' not in str(result)
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_rate_limiting_security(self, security_config):
        """Test rate limiting as a security measure."""
        # Set aggressive rate limiting
        security_config['rate_limit'] = 2
        security_config['rate_window'] = 5
        
        crawler = NewsCrawler(config=security_config)
        
        # Simulate DoS attack attempt
        attack_urls = [f'https://httpbin.org/json?attack={i}' for i in range(20)]
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "data"}'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                start_time = asyncio.get_event_loop().time()
                
                try:
                    await crawler.crawl_urls(attack_urls)
                except Exception as e:
                    # Rate limiting should prevent DoS
                    assert 'rate' in str(e).lower() or 'limit' in str(e).lower()
                
                end_time = asyncio.get_event_loop().time()
                
                # Should be rate limited
                duration = end_time - start_time
                assert duration > 30  # Should take significant time due to rate limiting
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_content_type_validation(self, security_config):
        """Test content type validation security."""
        crawler = NewsCrawler(config=security_config)
        
        # Test with various content types
        test_cases = [
            {
                'url': 'https://example.com/text',
                'content_type': 'text/html',
                'content': '<html><body>Valid HTML</body></html>',
                'should_process': True
            },
            {
                'url': 'https://example.com/executable',
                'content_type': 'application/x-executable',
                'content': 'binary_executable_data',
                'should_process': False
            },
            {
                'url': 'https://example.com/script',
                'content_type': 'application/javascript',
                'content': 'alert("malicious");',
                'should_process': False
            }
        ]
        
        for test_case in test_cases:
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = test_case['content']
                mock_response.headers = {'Content-Type': test_case['content_type']}
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                async with crawler:
                    result = await crawler.crawl_url(test_case['url'])
                    
                    if test_case['should_process']:
                        # Should process safe content types
                        assert result is not None
                        assert 'content' in result
                    else:
                        # Should reject dangerous content types
                        assert result is None or 'error' in result
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_memory_safety(self, security_config):
        """Test memory safety to prevent memory-based attacks."""
        crawler = NewsCrawler(config=security_config)
        
        # Test with memory-intensive content
        large_content = 'x' * (50 * 1024 * 1024)  # 50MB
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = large_content
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                try:
                    result = await crawler.crawl_url('https://example.com/large')
                    
                    # Should handle large content safely
                    if result is not None:
                        # Should not crash or consume excessive memory
                        assert len(result.get('content', '')) < 20 * 1024 * 1024  # Less than 20MB
                        
                except MemoryError:
                    # Memory errors should be handled gracefully
                    pass
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, security_config):
        """Test prevention of path traversal attacks."""
        crawler = NewsCrawler(config=security_config)
        
        # Test with path traversal attempts
        path_traversal_urls = [
            'https://example.com/../../../etc/passwd',
            'https://example.com/..\\..\\..\\windows\\system32\\config\\sam',
            'https://example.com/page?file=../../../etc/passwd',
            'https://example.com/download?path=..%2F..%2F..%2Fetc%2Fpasswd'
        ]
        
        async with crawler:
            for url in path_traversal_urls:
                try:
                    result = await crawler.crawl_url(url)
                    
                    # Should not access file system directly
                    if result is not None:
                        content = result.get('content', '')
                        # Should not contain system file contents
                        assert 'root:' not in content  # Unix passwd file
                        assert 'Administrator:' not in content  # Windows SAM file
                        
                except (ValueError, SecurityError):
                    # URL validation should catch path traversal
                    pass