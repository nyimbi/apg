"""
Gen Crawler Export Utilities
============================

Export utilities for converting crawl results to various formats
including clean markdown files, JSON, CSV, and HTML.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re
import html
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class BaseExporter:
    """Base class for all exporters."""
    
    def __init__(self, output_dir: Union[str, Path], **kwargs):
        """Initialize the exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = kwargs
    
    async def export_results(self, results: List[Any]) -> None:
        """Export results - to be implemented by subclasses."""
        raise NotImplementedError
    
    def _sanitize_filename(self, filename: str, max_length: int = 200) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename.strip('._')
        
        # Truncate if too long
        if len(filename) > max_length:
            filename = filename[:max_length-3] + '...'
        
        return filename or 'untitled'
    
    def _get_site_name(self, url: str) -> str:
        """Extract site name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            return domain.replace('www.', '').replace('.', '_')
        except:
            return 'unknown_site'

class MarkdownExporter(BaseExporter):
    """Export crawl results as clean markdown files."""
    
    def __init__(self, output_dir: Union[str, Path], **kwargs):
        """
        Initialize markdown exporter.
        
        Args:
            output_dir: Output directory
            save_images: Whether to save image references
            organize_by_site: Whether to organize files by site
            include_metadata: Whether to include metadata in markdown
            create_index: Whether to create index files
        """
        super().__init__(output_dir, **kwargs)
        self.save_images = kwargs.get('save_images', True)
        self.organize_by_site = kwargs.get('organize_by_site', True)
        self.include_metadata = kwargs.get('include_metadata', True)
        self.create_index = kwargs.get('create_index', True)
        self.organize_by = kwargs.get('organize_by', 'site')
    
    async def export_results(self, results: List[Any]) -> None:
        """Export results as markdown files."""
        
        logger.info(f"📝 Exporting {len(results)} sites to markdown")
        
        all_files = []
        
        for result in results:
            if not isinstance(result, dict) or 'pages' not in result:
                continue
            
            site_files = await self._export_site_to_markdown(result)
            all_files.extend(site_files)
        
        # Create index file if requested
        if self.create_index and all_files:
            await self._create_index_file(all_files, results)
        
        logger.info(f"✅ Exported {len(all_files)} markdown files")
    
    async def _export_site_to_markdown(self, site_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Export a single site's pages to markdown."""
        
        base_url = site_result.get('base_url', 'unknown')
        pages = site_result.get('pages', [])
        site_name = self._get_site_name(base_url)
        
        # Create site directory if organizing by site
        if self.organize_by_site:
            site_dir = self.output_dir / site_name
            site_dir.mkdir(exist_ok=True)
        else:
            site_dir = self.output_dir
        
        exported_files = []
        
        for i, page in enumerate(pages):
            if not isinstance(page, dict) or not page.get('success', False):
                continue
            
            try:
                # Generate markdown content
                markdown_content = await self._page_to_markdown(page, site_result)
                
                # Generate filename
                if self.organize_by == 'date' and page.get('timestamp'):
                    timestamp = datetime.fromisoformat(page['timestamp'].replace('Z', '+00:00'))
                    date_prefix = timestamp.strftime("%Y-%m-%d_")
                else:
                    date_prefix = ""
                
                title = page.get('title', '').strip()
                if not title:
                    title = f"page_{i:04d}"
                
                filename = f"{date_prefix}{self._sanitize_filename(title)}.md"
                filepath = site_dir / filename
                
                # Write markdown file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                exported_files.append({
                    'site': site_name,
                    'title': title,
                    'filename': filename,
                    'filepath': str(filepath),
                    'url': page.get('url', ''),
                    'word_count': page.get('word_count', 0),
                    'content_type': page.get('content_type', 'unknown'),
                    'quality_score': page.get('quality_score', 0.0)
                })
                
            except Exception as e:
                logger.warning(f"Failed to export page {page.get('url', 'unknown')}: {e}")
                continue
        
        return exported_files
    
    async def _page_to_markdown(self, page: Dict[str, Any], site_result: Dict[str, Any]) -> str:
        """Convert a page to markdown format."""
        
        markdown_lines = []
        
        # Title
        title = page.get('title', 'Untitled').strip()
        if title:
            markdown_lines.append(f"# {title}")
            markdown_lines.append("")
        
        # Metadata section
        if self.include_metadata:
            markdown_lines.append("## Metadata")
            markdown_lines.append("")
            
            # Basic metadata
            url = page.get('url', '')
            if url:
                markdown_lines.append(f"- **URL**: {url}")
            
            timestamp = page.get('timestamp')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                    markdown_lines.append(f"- **Crawled**: {formatted_date}")
                except:
                    markdown_lines.append(f"- **Crawled**: {timestamp}")
            
            content_type = page.get('content_type')
            if content_type:
                markdown_lines.append(f"- **Content Type**: {content_type}")
            
            word_count = page.get('word_count', 0)
            if word_count > 0:
                markdown_lines.append(f"- **Word Count**: {word_count:,}")
            
            quality_score = page.get('quality_score', 0.0)
            if quality_score > 0:
                markdown_lines.append(f"- **Quality Score**: {quality_score:.2f}")
            
            # Authors
            authors = page.get('authors', [])
            if authors:
                authors_str = ", ".join(authors)
                markdown_lines.append(f"- **Authors**: {authors_str}")
            
            # Keywords
            keywords = page.get('keywords', [])
            if keywords:
                keywords_str = ", ".join(keywords)
                markdown_lines.append(f"- **Keywords**: {keywords_str}")
            
            # Conflict analysis if available
            metadata = page.get('metadata', {})
            if metadata.get('conflict_related'):
                markdown_lines.append(f"- **Conflict Related**: Yes")
                conflict_keywords = metadata.get('conflict_keywords_found', [])
                if conflict_keywords:
                    markdown_lines.append(f"- **Conflict Keywords**: {', '.join(conflict_keywords)}")
            
            markdown_lines.append("")
        
        # Main content
        content = page.get('cleaned_content') or page.get('content', '')
        if content:
            markdown_lines.append("## Content")
            markdown_lines.append("")
            
            # Clean and format content
            cleaned_content = self._clean_content_for_markdown(content)
            markdown_lines.append(cleaned_content)
            markdown_lines.append("")
        
        # Summary if available
        summary = page.get('summary', '').strip()
        if summary:
            markdown_lines.append("## Summary")
            markdown_lines.append("")
            markdown_lines.append(summary)
            markdown_lines.append("")
        
        # Images section
        if self.save_images:
            images = page.get('images', [])
            if images:
                markdown_lines.append("## Images")
                markdown_lines.append("")
                for img_url in images[:10]:  # Limit to first 10 images
                    markdown_lines.append(f"![Image]({img_url})")
                markdown_lines.append("")
        
        # Links section (first 20 links)
        links = page.get('links', [])
        if links:
            markdown_lines.append("## Related Links")
            markdown_lines.append("")
            for link_url in links[:20]:
                try:
                    # Try to extract meaningful text from URL
                    link_text = urlparse(link_url).path.split('/')[-1] or link_url
                    markdown_lines.append(f"- [{link_text}]({link_url})")
                except:
                    markdown_lines.append(f"- {link_url}")
            markdown_lines.append("")
        
        # Footer
        markdown_lines.append("---")
        markdown_lines.append(f"*Crawled by Gen Crawler on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(markdown_lines)
    
    def _clean_content_for_markdown(self, content: str) -> str:
        """Clean content for markdown format."""
        
        # Decode HTML entities
        content = html.unescape(content)
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Split into paragraphs and clean
        paragraphs = content.split('\n\n')
        cleaned_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if len(para) > 20:  # Skip very short paragraphs
                # Escape markdown special characters in content
                para = re.sub(r'([*_`#])', r'\\\1', para)
                cleaned_paragraphs.append(para)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    async def _create_index_file(self, files: List[Dict[str, str]], results: List[Any]) -> None:
        """Create an index markdown file."""
        
        index_path = self.output_dir / "INDEX.md"
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# Crawl Results Index\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            total_files = len(files)
            total_sites = len(set(f['site'] for f in files))
            total_words = sum(f['word_count'] for f in files)
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Files**: {total_files:,}\n")
            f.write(f"- **Total Sites**: {total_sites}\n")
            f.write(f"- **Total Words**: {total_words:,}\n\n")
            
            # Files by site
            if self.organize_by_site:
                f.write("## Files by Site\n\n")
                
                sites = {}
                for file_info in files:
                    site = file_info['site']
                    if site not in sites:
                        sites[site] = []
                    sites[site].append(file_info)
                
                for site, site_files in sorted(sites.items()):
                    f.write(f"### {site}\n\n")
                    for file_info in sorted(site_files, key=lambda x: x['title']):
                        title = file_info['title']
                        filename = file_info['filename']
                        word_count = file_info['word_count']
                        quality = file_info['quality_score']
                        
                        f.write(f"- [{title}]({site}/{filename}) ")
                        f.write(f"({word_count:,} words, quality: {quality:.2f})\n")
                    f.write("\n")
            
            else:
                f.write("## All Files\n\n")
                for file_info in sorted(files, key=lambda x: x['title']):
                    title = file_info['title']
                    filename = file_info['filename']
                    site = file_info['site']
                    word_count = file_info['word_count']
                    
                    f.write(f"- [{title}]({filename}) ({site}, {word_count:,} words)\n")

class JSONExporter(BaseExporter):
    """Export crawl results as JSON files."""
    
    def __init__(self, output_dir: Union[str, Path], **kwargs):
        """
        Initialize JSON exporter.
        
        Args:
            output_dir: Output directory
            pretty_print: Whether to format JSON nicely
            compress: Whether to compress output
            separate_files: Whether to create separate files per site
        """
        super().__init__(output_dir, **kwargs)
        self.pretty_print = kwargs.get('pretty_print', True)
        self.compress = kwargs.get('compress', False)
        self.separate_files = kwargs.get('separate_files', False)
    
    async def export_results(self, results: List[Any]) -> None:
        """Export results as JSON."""
        
        logger.info(f"📄 Exporting {len(results)} sites to JSON")
        
        if self.separate_files:
            # Create separate JSON file for each site
            for i, result in enumerate(results):
                site_name = self._get_site_name(result.get('base_url', f'site_{i}'))
                filename = f"{site_name}_crawl_results.json"
                filepath = self.output_dir / filename
                
                await self._write_json_file(result, filepath)
        
        else:
            # Single combined JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crawl_results_{timestamp}.json"
            filepath = self.output_dir / filename
            
            combined_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_sites': len(results),
                'results': results
            }
            
            await self._write_json_file(combined_data, filepath)
        
        logger.info(f"✅ JSON export completed")
    
    async def _write_json_file(self, data: Any, filepath: Path) -> None:
        """Write data to JSON file."""
        
        try:
            json_kwargs = {'default': str}
            if self.pretty_print:
                json_kwargs.update({'indent': 2})
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, **json_kwargs)
            
            if self.compress:
                # Compress the file
                import gzip
                import shutil
                
                compressed_path = filepath.with_suffix(filepath.suffix + '.gz')
                with open(filepath, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original if compression successful
                filepath.unlink()
                
        except Exception as e:
            logger.error(f"Failed to write JSON file {filepath}: {e}")
            raise

class CSVExporter(BaseExporter):
    """Export crawl results as CSV files."""
    
    def __init__(self, output_dir: Union[str, Path], **kwargs):
        """Initialize CSV exporter."""
        super().__init__(output_dir, **kwargs)
        self.include_content = kwargs.get('include_content', False)
        self.max_content_length = kwargs.get('max_content_length', 1000)
    
    async def export_results(self, results: List[Any]) -> None:
        """Export results as CSV."""
        
        logger.info(f"📊 Exporting {len(results)} sites to CSV")
        
        # Flatten all pages into a single list
        all_pages = []
        
        for result in results:
            if not isinstance(result, dict) or 'pages' not in result:
                continue
            
            base_url = result.get('base_url', 'unknown')
            site_name = self._get_site_name(base_url)
            
            for page in result.get('pages', []):
                if isinstance(page, dict):
                    page_data = page.copy()
                    page_data['site_name'] = site_name
                    page_data['base_url'] = base_url
                    all_pages.append(page_data)
        
        if not all_pages:
            logger.warning("No pages to export")
            return
        
        # Create CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_dir / f"crawl_results_{timestamp}.csv"
        
        # Define CSV columns
        columns = [
            'site_name', 'base_url', 'url', 'title', 'content_type',
            'word_count', 'quality_score', 'success', 'crawl_time',
            'timestamp', 'authors', 'keywords'
        ]
        
        if self.include_content:
            columns.extend(['content', 'summary'])
        
        # Add conflict analysis columns if present
        if any(page.get('metadata', {}).get('conflict_related') for page in all_pages):
            columns.extend(['conflict_related', 'conflict_keywords_found'])
        
        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for page in all_pages:
                row = {}
                
                for col in columns:
                    if col in page:
                        value = page[col]
                        
                        # Handle list fields
                        if isinstance(value, list):
                            value = '; '.join(str(v) for v in value)
                        
                        # Truncate content if needed
                        elif col in ['content', 'summary'] and isinstance(value, str):
                            if len(value) > self.max_content_length:
                                value = value[:self.max_content_length] + '...'
                        
                        row[col] = value
                    
                    # Handle nested metadata
                    elif col.startswith('conflict_'):
                        metadata = page.get('metadata', {})
                        if col == 'conflict_related':
                            row[col] = metadata.get('conflict_related', False)
                        elif col == 'conflict_keywords_found':
                            keywords = metadata.get('conflict_keywords_found', [])
                            row[col] = '; '.join(keywords) if keywords else ''
                
                writer.writerow(row)
        
        logger.info(f"✅ CSV export completed: {csv_path}")

class HTMLExporter(BaseExporter):
    """Export crawl results as HTML files."""
    
    def __init__(self, output_dir: Union[str, Path], **kwargs):
        """Initialize HTML exporter."""
        super().__init__(output_dir, **kwargs)
        self.include_images = kwargs.get('include_images', True)
        self.create_navigation = kwargs.get('create_navigation', True)
    
    async def export_results(self, results: List[Any]) -> None:
        """Export results as HTML."""
        
        logger.info(f"🌐 Exporting {len(results)} sites to HTML")
        
        # Create CSS file
        await self._create_css_file()
        
        all_files = []
        
        for result in results:
            if not isinstance(result, dict) or 'pages' not in result:
                continue
            
            site_files = await self._export_site_to_html(result)
            all_files.extend(site_files)
        
        # Create index page
        if self.create_navigation:
            await self._create_index_html(all_files, results)
        
        logger.info(f"✅ HTML export completed: {len(all_files)} files")
    
    async def _export_site_to_html(self, site_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Export a site to HTML files."""
        
        base_url = site_result.get('base_url', 'unknown')
        pages = site_result.get('pages', [])
        site_name = self._get_site_name(base_url)
        
        site_dir = self.output_dir / site_name
        site_dir.mkdir(exist_ok=True)
        
        exported_files = []
        
        for i, page in enumerate(pages):
            if not isinstance(page, dict) or not page.get('success', False):
                continue
            
            try:
                html_content = await self._page_to_html(page)
                
                title = page.get('title', '').strip()
                if not title:
                    title = f"page_{i:04d}"
                
                filename = f"{self._sanitize_filename(title)}.html"
                filepath = site_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                exported_files.append({
                    'site': site_name,
                    'title': title,
                    'filename': filename,
                    'filepath': str(filepath),
                    'url': page.get('url', ''),
                    'word_count': page.get('word_count', 0)
                })
                
            except Exception as e:
                logger.warning(f"Failed to export HTML for {page.get('url', 'unknown')}: {e}")
                continue
        
        return exported_files
    
    async def _page_to_html(self, page: Dict[str, Any]) -> str:
        """Convert page to HTML."""
        
        title = html.escape(page.get('title', 'Untitled'))
        content = html.escape(page.get('cleaned_content') or page.get('content', ''))
        url = html.escape(page.get('url', ''))
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="../styles.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="metadata">
                <p><strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></p>
                <p><strong>Word Count:</strong> {page.get('word_count', 0):,}</p>
                <p><strong>Content Type:</strong> {page.get('content_type', 'unknown')}</p>
                <p><strong>Quality Score:</strong> {page.get('quality_score', 0.0):.2f}</p>
            </div>
        </header>
        
        <main>
            <div class="content">
                {self._format_content_for_html(content)}
            </div>
        </main>
        
        <footer>
            <p>Crawled by Gen Crawler on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
        """
        
        return html_template.strip()
    
    def _format_content_for_html(self, content: str) -> str:
        """Format content for HTML display."""
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        html_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if len(para) > 10:
                # Convert line breaks to HTML
                para = para.replace('\n', '<br>')
                html_paragraphs.append(f'<p>{para}</p>')
        
        return '\n'.join(html_paragraphs)
    
    async def _create_css_file(self) -> None:
        """Create CSS stylesheet."""
        
        css_content = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

.container {
    background: white;
    padding: 40px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}

.metadata {
    background: #ecf0f1;
    padding: 15px;
    border-radius: 5px;
    margin: 20px 0;
}

.metadata p {
    margin: 5px 0;
}

.content {
    margin: 30px 0;
}

.content p {
    margin-bottom: 15px;
    text-align: justify;
}

footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #eee;
    color: #666;
    font-size: 0.9em;
}

a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

.index-container {
    max-width: 1200px;
}

.site-section {
    margin-bottom: 40px;
}

.file-list {
    list-style: none;
    padding: 0;
}

.file-list li {
    padding: 10px;
    border-bottom: 1px solid #eee;
}

.file-list li:hover {
    background-color: #f8f9fa;
}
        """
        
        css_path = self.output_dir / "styles.css"
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
    
    async def _create_index_html(self, files: List[Dict[str, str]], results: List[Any]) -> None:
        """Create index HTML file."""
        
        index_path = self.output_dir / "index.html"
        
        # Group files by site
        sites = {}
        for file_info in files:
            site = file_info['site']
            if site not in sites:
                sites[site] = []
            sites[site].append(file_info)
        
        # Build HTML
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crawl Results Index</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container index-container">
        <header>
            <h1>Crawl Results Index</h1>
            <div class="metadata">
                <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                <p><strong>Total Files:</strong> """ + f"{len(files):,}" + """</p>
                <p><strong>Total Sites:</strong> """ + str(len(sites)) + """</p>
            </div>
        </header>
        
        <main>
        """
        
        for site, site_files in sorted(sites.items()):
            html_content += f"""
            <div class="site-section">
                <h2>{html.escape(site)}</h2>
                <ul class="file-list">
            """
            
            for file_info in sorted(site_files, key=lambda x: x['title']):
                title = html.escape(file_info['title'])
                filename = file_info['filename']
                word_count = file_info['word_count']
                
                html_content += f"""
                    <li>
                        <a href="{site}/{filename}">{title}</a>
                        <span style="color: #666; margin-left: 10px;">
                            ({word_count:,} words)
                        </span>
                    </li>
                """
            
            html_content += """
                </ul>
            </div>
            """
        
        html_content += """
        </main>
        
        <footer>
            <p>Generated by Gen Crawler</p>
        </footer>
    </div>
</body>
</html>
        """
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)