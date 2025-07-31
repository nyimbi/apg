"""
Content Parser Tests
===================

Test suite for gen_crawler content parsing and analysis components.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

try:
    from ..parsers.content_parser import (
        GenContentParser, ParsedSiteContent, ContentAnalyzer,
        create_content_parser
    )
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False

if not PARSERS_AVAILABLE:
    import pytest
    pytest.skip("Parser components not available", allow_module_level=True)

class TestParsedSiteContent(unittest.TestCase):
    """Test ParsedSiteContent data class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_content = ParsedSiteContent(
            url="https://example.com/article",
            title="Test Article",
            content="This is a test article with substantial content for testing purposes.",
            word_count=12,
            quality_score=0.8,
            content_type="article"
        )
    
    def test_content_creation(self):
        """Test basic content creation."""
        self.assertEqual(self.sample_content.url, "https://example.com/article")
        self.assertEqual(self.sample_content.title, "Test Article")
        self.assertEqual(self.sample_content.word_count, 12)
        self.assertEqual(self.sample_content.quality_score, 0.8)
        self.assertEqual(self.sample_content.content_type, "article")
    
    def test_is_article_property(self):
        """Test is_article property logic."""
        # Test with article content type
        self.assertTrue(self.sample_content.is_article)
        
        # Test with sufficient word count and title
        content = ParsedSiteContent(
            url="https://example.com/test",
            title="Long Title Here",
            content="Content " * 100,
            word_count=350,
            content_type="unknown"
        )
        self.assertTrue(content.is_article)
        
        # Test with insufficient content
        short_content = ParsedSiteContent(
            url="https://example.com/short",
            title="Short",
            content="Short content",
            word_count=2,
            content_type="snippet"
        )
        self.assertFalse(short_content.is_article)
    
    def test_is_high_quality_property(self):
        """Test is_high_quality property logic."""
        # Test high quality content
        self.assertTrue(self.sample_content.is_high_quality)
        
        # Test low quality content
        low_quality = ParsedSiteContent(
            url="https://example.com/low",
            title="",
            content="Short",
            word_count=1,
            quality_score=0.3,
            content_type="snippet"
        )
        self.assertFalse(low_quality.is_high_quality)
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        content_dict = self.sample_content.to_dict()
        
        self.assertIsInstance(content_dict, dict)
        self.assertEqual(content_dict['url'], "https://example.com/article")
        self.assertEqual(content_dict['title'], "Test Article")
        self.assertEqual(content_dict['word_count'], 12)
        self.assertEqual(content_dict['quality_score'], 0.8)
        self.assertIn('parse_timestamp', content_dict)
        self.assertTrue(content_dict['is_article'])
        self.assertTrue(content_dict['is_high_quality'])

class TestContentAnalyzer(unittest.TestCase):
    """Test ContentAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ContentAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer.article_indicators, list)
        self.assertIsInstance(self.analyzer.quality_indicators, list)
        self.assertIsInstance(self.analyzer.low_quality_indicators, list)
        
        self.assertIn('article', self.analyzer.article_indicators)
        self.assertIn('byline', self.analyzer.quality_indicators)
        self.assertIn('advertisement', self.analyzer.low_quality_indicators)
    
    def test_content_type_analysis_article_url(self):
        """Test content type analysis based on URL."""
        # Article URL patterns
        article_url = "https://example.com/news/article/breaking-story"
        content_type = self.analyzer.analyze_content_type(
            article_url, "Breaking Story", "This is article content.", ""
        )
        self.assertEqual(content_type, "article")
        
        # Blog post URL
        blog_url = "https://example.com/blog/post/my-thoughts"
        content_type = self.analyzer.analyze_content_type(
            blog_url, "My Thoughts", "Blog post content.", ""
        )
        self.assertEqual(content_type, "article")
    
    def test_content_type_analysis_title(self):
        """Test content type analysis based on title."""
        # Article title
        content_type = self.analyzer.analyze_content_type(
            "https://example.com/page",
            "Breaking News: Important Story",
            "News content here.",
            ""
        )
        self.assertEqual(content_type, "article")
        
        # Report title
        content_type = self.analyzer.analyze_content_type(
            "https://example.com/page",
            "Annual Report on Climate Change",
            "Report content here.",
            ""
        )
        self.assertEqual(content_type, "article")
    
    def test_content_type_analysis_length(self):
        """Test content type analysis based on content length."""
        # Long content should be classified as article
        long_content = "This is a very long article. " * 50
        content_type = self.analyzer.analyze_content_type(
            "https://example.com/page",
            "Some Title",
            long_content,
            ""
        )
        self.assertEqual(content_type, "article")
        
        # Medium content
        medium_content = "This is medium length content. " * 10
        content_type = self.analyzer.analyze_content_type(
            "https://example.com/page",
            "Some Title",
            medium_content,
            ""
        )
        self.assertEqual(content_type, "content_page")
        
        # Short content
        short_content = "Short content here."
        content_type = self.analyzer.analyze_content_type(
            "https://example.com/page",
            "Some Title",
            short_content,
            ""
        )
        self.assertEqual(content_type, "snippet")
    
    def test_content_type_analysis_special_pages(self):
        """Test content type analysis for special page types."""
        # Category page
        category_type = self.analyzer.analyze_content_type(
            "https://example.com/category/sports",
            "Sports Category",
            "Sports articles list...",
            ""
        )
        self.assertEqual(category_type, "listing")
        
        # About page
        about_type = self.analyzer.analyze_content_type(
            "https://example.com/about",
            "About Us",
            "Information about our company...",
            ""
        )
        self.assertEqual(about_type, "page")
        
        # Insufficient content
        insufficient_type = self.analyzer.analyze_content_type(
            "https://example.com/empty",
            "",
            "Short",
            ""
        )
        self.assertEqual(insufficient_type, "insufficient_content")
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # High quality content
        high_quality_content = ParsedSiteContent(
            url="https://example.com/article",
            title="Comprehensive Analysis of Climate Change Impacts",
            content="This is a detailed article with substantial content. " * 100,
            cleaned_content="Clean content...",
            word_count=800,
            authors=["Dr. Jane Smith"],
            keywords=["climate", "environment", "research"],
            metadata={"category": "science", "publish_date": "2025-01-01"}
        )
        
        score = self.analyzer.calculate_quality_score(high_quality_content, "")
        self.assertGreater(score, 0.7)
        
        # Low quality content
        low_quality_content = ParsedSiteContent(
            url="https://example.com/snippet",
            title="",
            content="Short content",
            word_count=2,
            authors=[],
            keywords=[],
            metadata={}
        )
        
        score = self.analyzer.calculate_quality_score(low_quality_content, "")
        self.assertLess(score, 0.3)
    
    def test_quality_score_components(self):
        """Test individual components of quality score."""
        # Test title component
        good_title = ParsedSiteContent(
            url="https://example.com/article",
            title="This is a well-written, comprehensive title about the topic",
            content="Content here",
            word_count=10
        )
        
        no_title = ParsedSiteContent(
            url="https://example.com/article",
            title="",
            content="Content here",
            word_count=10
        )
        
        score_with_title = self.analyzer.calculate_quality_score(good_title, "")
        score_without_title = self.analyzer.calculate_quality_score(no_title, "")
        
        self.assertGreater(score_with_title, score_without_title)
    
    def test_quality_score_with_html(self):
        """Test quality score calculation with HTML analysis."""
        content = ParsedSiteContent(
            url="https://example.com/article",
            title="Test Article",
            content="Test content",
            word_count=10
        )
        
        # HTML with quality indicators
        quality_html = """
        <html>
        <head><meta name="author" content="John Doe"></head>
        <body>
            <article>
                <div class="byline">By John Doe</div>
                <time datetime="2025-01-01">January 1, 2025</time>
                <p>Article content...</p>
            </article>
        </body>
        </html>
        """
        
        # HTML with low quality indicators
        low_quality_html = """
        <html>
        <body>
            <div class="advertisement">Ad content</div>
            <div class="popup">Subscribe now!</div>
            <div class="sponsored">Sponsored content</div>
        </body>
        </html>
        """
        
        quality_score = self.analyzer.calculate_quality_score(content, quality_html)
        low_quality_score = self.analyzer.calculate_quality_score(content, low_quality_html)
        
        self.assertGreater(quality_score, low_quality_score)

class TestGenContentParser(unittest.TestCase):
    """Test GenContentParser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = GenContentParser()
        
        self.sample_html = """
        <html>
        <head>
            <title>Test Article Title</title>
            <meta name="author" content="John Doe">
            <meta name="description" content="Article description">
            <meta name="keywords" content="test, article, sample">
        </head>
        <body>
            <script>console.log('test');</script>
            <style>body { margin: 0; }</style>
            <article>
                <h1>Test Article Title</h1>
                <p>This is the first paragraph of the test article.</p>
                <p>This is the second paragraph with more content.</p>
                <p>Final paragraph to complete the article content.</p>
            </article>
        </body>
        </html>
        """
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        self.assertIsInstance(self.parser.extraction_methods, list)
        self.assertIsNotNone(self.parser.analyzer)
        
        # Check available methods (at least basic should be available)
        self.assertGreater(len(self.parser.extraction_methods), 0)
    
    def test_parser_status(self):
        """Test parser status reporting."""
        status = self.parser.get_parser_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('available_methods', status)
        self.assertIn('total_methods', status)
        self.assertIn('beautifulsoup_available', status)
        
        self.assertGreaterEqual(status['total_methods'], 1)  # At least basic parsing
    
    def test_parse_content_basic(self):
        """Test basic content parsing."""
        url = "https://example.com/article"
        
        parsed = self.parser.parse_content(url, self.sample_html)
        
        self.assertIsInstance(parsed, ParsedSiteContent)
        self.assertEqual(parsed.url, url)
        self.assertIn("Test Article Title", parsed.title)
        self.assertIn("first paragraph", parsed.content)
        self.assertGreater(parsed.word_count, 0)
        self.assertGreater(parsed.quality_score, 0)
    
    def test_parse_empty_content(self):
        """Test parsing empty content."""
        url = "https://example.com/empty"
        
        parsed = self.parser.parse_content(url, "")
        
        self.assertEqual(parsed.url, url)
        self.assertEqual(parsed.extraction_method, "none")
        self.assertEqual(parsed.word_count, 0)
    
    def test_parse_content_with_preferred_method(self):
        """Test parsing with preferred extraction method."""
        url = "https://example.com/article"
        
        # Test with beautifulsoup (should be available)
        parsed = self.parser.parse_content(url, self.sample_html, "beautifulsoup")
        
        if "beautifulsoup" in self.parser.extraction_methods:
            self.assertEqual(parsed.extraction_method, "beautifulsoup")
        else:
            # Should fall back to available method
            self.assertIn(parsed.extraction_method, self.parser.extraction_methods + ["basic"])
    
    @patch('gen_crawler.parsers.content_parser.TRAFILATURA_AVAILABLE', True)
    @patch('gen_crawler.parsers.content_parser.trafilatura')
    def test_trafilatura_extraction(self, mock_trafilatura):
        """Test Trafilatura extraction method."""
        # Mock Trafilatura response
        mock_trafilatura.extract.return_value = "Extracted content from Trafilatura"
        mock_metadata = MagicMock()
        mock_metadata.title = "Trafilatura Title"
        mock_metadata.author = "Test Author"
        mock_metadata.date = datetime.now()
        mock_metadata.language = "en"
        mock_metadata.tags = ["tag1", "tag2"]
        mock_trafilatura.extract_metadata.return_value = mock_metadata
        
        # Force trafilatura to be in extraction methods
        parser = GenContentParser()
        parser.extraction_methods = ['trafilatura']
        
        url = "https://example.com/article"
        parsed = parser.parse_content(url, self.sample_html, "trafilatura")
        
        self.assertEqual(parsed.extraction_method, "trafilatura")
        self.assertEqual(parsed.content, "Extracted content from Trafilatura")
        self.assertEqual(parsed.title, "Trafilatura Title")
        self.assertEqual(parsed.authors, ["Test Author"])
        self.assertEqual(parsed.keywords, ["tag1", "tag2"])
    
    @patch('gen_crawler.parsers.content_parser.NEWSPAPER_AVAILABLE', True)
    @patch('gen_crawler.parsers.content_parser.Article')
    def test_newspaper_extraction(self, mock_article_class):
        """Test Newspaper3k extraction method."""
        # Mock Newspaper3k response
        mock_article = MagicMock()
        mock_article.title = "Newspaper Title"
        mock_article.text = "Extracted content from Newspaper3k"
        mock_article.authors = ["Author One", "Author Two"]
        mock_article.publish_date = datetime.now()
        mock_article.keywords = ["keyword1", "keyword2"]
        mock_article.summary = "Article summary"
        mock_article.meta_data = {"description": "Meta description"}
        
        mock_article_class.return_value = mock_article
        
        # Force newspaper to be in extraction methods
        parser = GenContentParser()
        parser.extraction_methods = ['newspaper']
        
        url = "https://example.com/article"
        parsed = parser.parse_content(url, self.sample_html, "newspaper")
        
        self.assertEqual(parsed.extraction_method, "newspaper")
        self.assertEqual(parsed.content, "Extracted content from Newspaper3k")
        self.assertEqual(parsed.title, "Newspaper Title")
        self.assertEqual(parsed.authors, ["Author One", "Author Two"])
        self.assertEqual(parsed.summary, "Article summary")
    
    def test_beautifulsoup_extraction(self):
        """Test BeautifulSoup extraction method."""
        # BeautifulSoup should be available in most environments
        if "beautifulsoup" not in self.parser.extraction_methods:
            self.skipTest("BeautifulSoup not available")
        
        url = "https://example.com/article"
        parsed = self.parser.parse_content(url, self.sample_html, "beautifulsoup")
        
        self.assertEqual(parsed.extraction_method, "beautifulsoup")
        self.assertIn("Test Article Title", parsed.title)
        self.assertIn("first paragraph", parsed.content)
        
        # Should remove script and style content
        self.assertNotIn("console.log", parsed.content)
        self.assertNotIn("margin: 0", parsed.content)
    
    def test_basic_fallback_extraction(self):
        """Test basic fallback extraction method."""
        # Create parser with no extraction methods to force fallback
        parser = GenContentParser()
        parser.extraction_methods = []
        
        url = "https://example.com/article"
        parsed = parser.parse_content(url, self.sample_html)
        
        self.assertEqual(parsed.extraction_method, "basic")
        self.assertIn("Test Article Title", parsed.title)
        self.assertIn("paragraph", parsed.content)
        
        # Should remove script and style content
        self.assertNotIn("console.log", parsed.content)
        self.assertNotIn("margin: 0", parsed.content)
    
    def test_content_cleaning(self):
        """Test content cleaning functionality."""
        dirty_html = """
        <html>
        <head><title>Test &amp; Article</title></head>
        <body>
            <p>First    paragraph   with    extra    spaces.</p>
            
            
            <p>Second paragraph after many newlines.</p>
            <p>Third paragraph with *asterisks* and _underscores_.</p>
        </body>
        </html>
        """
        
        url = "https://example.com/article"
        parsed = self.parser.parse_content(url, dirty_html)
        
        # Should decode HTML entities
        self.assertIn("Test & Article", parsed.title)
        
        # Should clean up excessive whitespace
        self.assertNotIn("    ", parsed.content)
        self.assertNotIn("\n\n\n", parsed.content)

class TestFactoryFunction(unittest.TestCase):
    """Test factory function for content parser."""
    
    def test_create_content_parser(self):
        """Test create_content_parser factory function."""
        parser = create_content_parser()
        
        self.assertIsInstance(parser, GenContentParser)
        self.assertIsNotNone(parser.analyzer)
    
    def test_create_content_parser_with_config(self):
        """Test create_content_parser with configuration."""
        config = {
            'preferred_method': 'beautifulsoup',
            'quality_threshold': 0.8
        }
        
        parser = create_content_parser(config)
        
        self.assertIsInstance(parser, GenContentParser)
        self.assertEqual(parser.config, config)

class TestParserIntegration(unittest.TestCase):
    """Integration tests for parser components."""
    
    def test_full_parsing_workflow(self):
        """Test complete parsing workflow."""
        parser = GenContentParser()
        analyzer = ContentAnalyzer()
        
        # Sample HTML with various content types
        samples = [
            {
                'html': """
                <html>
                <head><title>Breaking News: Major Event</title></head>
                <body><article><p>""" + "News content. " * 100 + """</p></article></body>
                </html>
                """,
                'expected_type': 'article'
            },
            {
                'html': """
                <html>
                <head><title>About Our Company</title></head>
                <body><p>""" + "Company information. " * 20 + """</p></body>
                </html>
                """,
                'expected_type': 'page'
            },
            {
                'html': """
                <html>
                <head><title>Quick Note</title></head>
                <body><p>Short content here.</p></body>
                </html>
                """,
                'expected_type': 'snippet'
            }
        ]
        
        for i, sample in enumerate(samples):
            url = f"https://example.com/page{i}"
            parsed = parser.parse_content(url, sample['html'])
            
            # Verify parsing worked
            self.assertGreater(len(parsed.content), 0)
            self.assertGreater(parsed.word_count, 0)
            
            # Verify content type classification
            content_type = analyzer.analyze_content_type(
                url, parsed.title, parsed.content, sample['html']
            )
            
            if sample['expected_type'] == 'article':
                self.assertIn(content_type, ['article', 'content_page'])
            else:
                self.assertEqual(content_type, sample['expected_type'])
    
    def test_quality_scoring_consistency(self):
        """Test consistency of quality scoring."""
        parser = GenContentParser()
        
        # High quality content
        high_quality_html = """
        <html>
        <head>
            <title>Comprehensive Guide to Climate Change Research Methods</title>
            <meta name="author" content="Dr. Jane Smith">
            <meta name="keywords" content="climate, research, methodology">
        </head>
        <body>
            <article>
                <h1>Comprehensive Guide to Climate Change Research Methods</h1>
                """ + "<p>Detailed paragraph about research methods. </p>" * 50 + """
            </article>
        </body>
        </html>
        """
        
        # Low quality content
        low_quality_html = """
        <html>
        <head><title></title></head>
        <body>
            <div class="advertisement">Buy now!</div>
            <p>Short ad content.</p>
        </body>
        </html>
        """
        
        high_quality_parsed = parser.parse_content("https://example.com/good", high_quality_html)
        low_quality_parsed = parser.parse_content("https://example.com/bad", low_quality_html)
        
        self.assertGreater(high_quality_parsed.quality_score, low_quality_parsed.quality_score)
        self.assertTrue(high_quality_parsed.is_high_quality)
        self.assertFalse(low_quality_parsed.is_high_quality)

if __name__ == '__main__':
    unittest.main(verbosity=2)