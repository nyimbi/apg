"""
Thumbnail Parser Module
=======================

Specialized parsers for YouTube thumbnail image data extraction and analysis.
Handles thumbnail processing, image analysis, and quality assessment.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import re
import time
import base64
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from io import BytesIO

from .base_parser import BaseParser, ParseResult, ParseStatus, ContentType
from ..api.data_models import ThumbnailData

logger = logging.getLogger(__name__)

# Optional dependencies for image processing
try:
    from PIL import Image, ImageStat
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ThumbnailParser(BaseParser):
    """Main thumbnail parser for image data extraction."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processor = ThumbnailProcessor(**kwargs)
        self.analyzer = ThumbnailAnalyzer(**kwargs)
        
        # Setup HTTP session for downloading
        if HAS_REQUESTS:
            self.session = requests.Session()
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
    
    async def parse(self, data: Union[Dict, str, bytes], content_type: ContentType) -> ParseResult:
        """Parse thumbnail data from various sources."""
        start_time = time.time()
        
        try:
            thumbnail_data = None
            
            if isinstance(data, str):
                if data.startswith('http'):
                    # URL to thumbnail
                    thumbnail_data = await self._parse_from_url(data)
                elif data.startswith('data:image'):
                    # Base64 data URL
                    thumbnail_data = await self._parse_from_data_url(data)
                else:
                    # Assume it's a local file path or JSON string
                    try:
                        import json
                        parsed_data = json.loads(data)
                        thumbnail_data = await self._parse_from_dict(parsed_data)
                    except:
                        # Treat as file path
                        thumbnail_data = await self._parse_from_file(data)
            elif isinstance(data, bytes):
                # Raw image bytes
                thumbnail_data = await self._parse_from_bytes(data)
            elif isinstance(data, dict):
                # Structured thumbnail data
                thumbnail_data = await self._parse_from_dict(data)
            
            if not thumbnail_data:
                raise ValueError("No valid thumbnail data found")
            
            # Process image if requested
            if self.config.process_images and thumbnail_data:
                process_result = await self.processor.process_thumbnail(thumbnail_data)
                if process_result.is_successful():
                    processed_data = process_result.data
                    if processed_data:
                        # Update thumbnail with processed data
                        thumbnail_data.file_size = processed_data.get('file_size')
                        thumbnail_data.compression_ratio = processed_data.get('compression_ratio')
                        thumbnail_data.dpi = processed_data.get('dpi')
            
            # Analyze image if requested
            if self.config.analyze_content and thumbnail_data:
                analysis_result = await self.analyzer.analyze_thumbnail(thumbnail_data)
                if analysis_result.is_successful():
                    analysis_data = analysis_result.data
                    if analysis_data:
                        thumbnail_data.dominant_colors = analysis_data.get('dominant_colors', [])
                        thumbnail_data.has_text = analysis_data.get('has_text', False)
                        thumbnail_data.has_faces = analysis_data.get('has_faces', False)
                        thumbnail_data.quality_score = analysis_data.get('quality_score', 0.0)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=thumbnail_data,
                execution_time=execution_time,
                metadata={
                    'parser': 'ThumbnailParser',
                    'source': 'url' if isinstance(data, str) and data.startswith('http') else 'data',
                    'format': thumbnail_data.format if thumbnail_data else 'unknown'
                }
            )
            
        except Exception as e:
            logger.error(f"Thumbnail parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _parse_from_url(self, url: str) -> Optional[ThumbnailData]:
        """Parse thumbnail from URL."""
        if not HAS_REQUESTS:
            logger.warning("requests library not available for URL parsing")
            return self._create_basic_thumbnail_from_url(url)
        
        try:
            # Download image
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            image_data = response.content
            content_type = response.headers.get('content-type', 'image/jpeg')
            
            # Determine format
            format_type = 'jpg'
            if 'png' in content_type:
                format_type = 'png'
            elif 'webp' in content_type:
                format_type = 'webp'
            elif 'gif' in content_type:
                format_type = 'gif'
            
            # Get image dimensions if PIL is available
            width, height = 0, 0
            if HAS_PIL:
                try:
                    img = Image.open(BytesIO(image_data))
                    width, height = img.size
                except Exception as e:
                    logger.warning(f"Could not get image dimensions: {e}")
            
            # Extract video ID from URL (YouTube thumbnail URLs)
            video_id = self._extract_video_id_from_thumbnail_url(url)
            
            # Determine size name from URL
            size_name = self._determine_size_name_from_url(url)
            
            return ThumbnailData(
                video_id=video_id or "unknown",
                url=url,
                width=width,
                height=height,
                size_name=size_name,
                file_size=len(image_data),
                format=format_type,
                binary_data=image_data if self.config.store_binary_data else None
            )
            
        except Exception as e:
            logger.error(f"Failed to download thumbnail from URL {url}: {e}")
            return self._create_basic_thumbnail_from_url(url)
    
    async def _parse_from_data_url(self, data_url: str) -> Optional[ThumbnailData]:
        """Parse thumbnail from base64 data URL."""
        try:
            # Extract metadata and data from data URL
            header, encoded = data_url.split(',', 1)
            metadata = header.split(';')
            
            # Get MIME type
            mime_type = metadata[0].split(':')[1] if ':' in metadata[0] else 'image/jpeg'
            format_type = mime_type.split('/')[-1]
            
            # Decode image data
            image_data = base64.b64decode(encoded)
            
            # Get image dimensions if PIL is available
            width, height = 0, 0
            if HAS_PIL:
                try:
                    img = Image.open(BytesIO(image_data))
                    width, height = img.size
                except Exception as e:
                    logger.warning(f"Could not get image dimensions: {e}")
            
            return ThumbnailData(
                video_id="unknown",
                url=data_url,
                width=width,
                height=height,
                size_name="unknown",
                file_size=len(image_data),
                format=format_type,
                binary_data=image_data if self.config.store_binary_data else None
            )
            
        except Exception as e:
            logger.error(f"Failed to parse data URL: {e}")
            return None
    
    async def _parse_from_file(self, file_path: str) -> Optional[ThumbnailData]:
        """Parse thumbnail from file path."""
        try:
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # Determine format from file extension
            format_type = file_path.split('.')[-1].lower()
            if format_type == 'jpeg':
                format_type = 'jpg'
            
            # Get image dimensions if PIL is available
            width, height = 0, 0
            if HAS_PIL:
                try:
                    img = Image.open(file_path)
                    width, height = img.size
                except Exception as e:
                    logger.warning(f"Could not get image dimensions: {e}")
            
            return ThumbnailData(
                video_id="unknown",
                url=f"file://{file_path}",
                width=width,
                height=height,
                size_name="unknown",
                file_size=len(image_data),
                format=format_type,
                binary_data=image_data if self.config.store_binary_data else None
            )
            
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return None
    
    async def _parse_from_bytes(self, image_data: bytes) -> Optional[ThumbnailData]:
        """Parse thumbnail from raw bytes."""
        try:
            # Try to determine format from image header
            format_type = self._detect_image_format(image_data)
            
            # Get image dimensions if PIL is available
            width, height = 0, 0
            if HAS_PIL:
                try:
                    img = Image.open(BytesIO(image_data))
                    width, height = img.size
                    if not format_type:
                        format_type = img.format.lower()
                except Exception as e:
                    logger.warning(f"Could not get image dimensions: {e}")
            
            return ThumbnailData(
                video_id="unknown",
                url="data:image",
                width=width,
                height=height,
                size_name="unknown",
                file_size=len(image_data),
                format=format_type or 'jpg',
                binary_data=image_data if self.config.store_binary_data else None
            )
            
        except Exception as e:
            logger.error(f"Failed to parse image bytes: {e}")
            return None
    
    async def _parse_from_dict(self, data: Dict[str, Any]) -> Optional[ThumbnailData]:
        """Parse thumbnail from dictionary data."""
        try:
            return ThumbnailData(
                video_id=data.get('video_id', 'unknown'),
                url=data.get('url', ''),
                width=int(data.get('width', 0)),
                height=int(data.get('height', 0)),
                size_name=data.get('size_name', data.get('size', 'unknown')),
                file_size=data.get('file_size'),
                format=data.get('format', 'jpg'),
                quality_score=float(data.get('quality_score', 0.0)),
                dominant_colors=data.get('dominant_colors', []),
                has_text=data.get('has_text', False),
                has_faces=data.get('has_faces', False),
                compression_ratio=data.get('compression_ratio'),
                dpi=data.get('dpi')
            )
            
        except Exception as e:
            logger.error(f"Failed to parse thumbnail dict: {e}")
            return None
    
    def _create_basic_thumbnail_from_url(self, url: str) -> ThumbnailData:
        """Create basic thumbnail data from URL when download fails."""
        video_id = self._extract_video_id_from_thumbnail_url(url)
        size_name = self._determine_size_name_from_url(url)
        
        # Guess dimensions based on size name
        width, height = self._guess_dimensions_from_size(size_name)
        
        return ThumbnailData(
            video_id=video_id or "unknown",
            url=url,
            width=width,
            height=height,
            size_name=size_name,
            format='jpg'
        )
    
    def _extract_video_id_from_thumbnail_url(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube thumbnail URL."""
        patterns = [
            r'vi/([^/]+)/',
            r'/vi/([^/]+)/',
            r'img.youtube.com/vi/([^/]+)/',
            r'i.ytimg.com/vi/([^/]+)/',
            r'i1.ytimg.com/vi/([^/]+)/',
            r'i2.ytimg.com/vi/([^/]+)/',
            r'i3.ytimg.com/vi/([^/]+)/',
            r'i4.ytimg.com/vi/([^/]+)/'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _determine_size_name_from_url(self, url: str) -> str:
        """Determine size name from thumbnail URL."""
        if 'maxresdefault' in url:
            return 'maxres'
        elif 'hqdefault' in url:
            return 'high'
        elif 'mqdefault' in url:
            return 'medium'
        elif 'sddefault' in url:
            return 'standard'
        elif 'default' in url:
            return 'default'
        else:
            return 'unknown'
    
    def _guess_dimensions_from_size(self, size_name: str) -> Tuple[int, int]:
        """Guess thumbnail dimensions based on size name."""
        size_map = {
            'maxres': (1280, 720),
            'high': (480, 360),
            'medium': (320, 180),
            'standard': (640, 480),
            'default': (120, 90)
        }
        return size_map.get(size_name, (0, 0))
    
    def _detect_image_format(self, image_data: bytes) -> Optional[str]:
        """Detect image format from binary data."""
        if image_data.startswith(b'\xff\xd8\xff'):
            return 'jpg'
        elif image_data.startswith(b'\x89\x50\x4e\x47'):
            return 'png'
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
            return 'webp'
        elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
            return 'gif'
        else:
            return None


class ThumbnailProcessor(BaseParser):
    """Processor for thumbnail image operations."""
    
    async def process_thumbnail(self, thumbnail: ThumbnailData) -> ParseResult:
        """Process thumbnail image data."""
        start_time = time.time()
        
        try:
            processing_data = {}
            
            if not thumbnail.binary_data:
                # Try to download if URL is available
                if thumbnail.url and thumbnail.url.startswith('http'):
                    if HAS_REQUESTS:
                        try:
                            response = requests.get(thumbnail.url, timeout=10)
                            thumbnail.binary_data = response.content
                        except Exception as e:
                            logger.warning(f"Could not download thumbnail for processing: {e}")
            
            if thumbnail.binary_data and HAS_PIL:
                try:
                    img = Image.open(BytesIO(thumbnail.binary_data))
                    
                    # File size
                    processing_data['file_size'] = len(thumbnail.binary_data)
                    
                    # Calculate compression ratio (simplified)
                    uncompressed_size = img.width * img.height * 3  # Assuming RGB
                    if uncompressed_size > 0:
                        processing_data['compression_ratio'] = len(thumbnail.binary_data) / uncompressed_size
                    
                    # DPI if available
                    dpi = img.info.get('dpi')
                    if dpi:
                        processing_data['dpi'] = dpi[0] if isinstance(dpi, tuple) else dpi
                    
                    # Additional image stats
                    stat = ImageStat.Stat(img)
                    processing_data['mean_brightness'] = stat.mean[0] if stat.mean else 0
                    processing_data['std_deviation'] = stat.stddev[0] if stat.stddev else 0
                    
                except Exception as e:
                    logger.warning(f"PIL processing failed: {e}")
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=processing_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Thumbnail processing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )


class ThumbnailAnalyzer(BaseParser):
    """Analyzer for thumbnail image content analysis."""
    
    async def analyze_thumbnail(self, thumbnail: ThumbnailData) -> ParseResult:
        """Analyze thumbnail image content."""
        start_time = time.time()
        
        try:
            analysis_data = {
                'dominant_colors': [],
                'has_text': False,
                'has_faces': False,
                'quality_score': 0.0,
                'brightness_score': 0.0,
                'contrast_score': 0.0,
                'color_distribution': {},
                'composition_score': 0.0
            }
            
            if not thumbnail.binary_data:
                # Try to download if URL is available
                if thumbnail.url and thumbnail.url.startswith('http'):
                    if HAS_REQUESTS:
                        try:
                            response = requests.get(thumbnail.url, timeout=10)
                            thumbnail.binary_data = response.content
                        except Exception as e:
                            logger.warning(f"Could not download thumbnail for analysis: {e}")
            
            if thumbnail.binary_data:
                # PIL-based analysis
                if HAS_PIL:
                    analysis_data.update(await self._analyze_with_pil(thumbnail.binary_data))
                
                # OpenCV-based analysis
                if HAS_OPENCV:
                    opencv_analysis = await self._analyze_with_opencv(thumbnail.binary_data)
                    analysis_data.update(opencv_analysis)
            
            # Calculate overall quality score
            analysis_data['quality_score'] = self._calculate_quality_score(analysis_data, thumbnail)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=analysis_data,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Thumbnail analysis failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _analyze_with_pil(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze image using PIL."""
        analysis = {}
        
        try:
            img = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get dominant colors
            colors = img.getcolors(maxcolors=256*256*256)
            if colors:
                # Sort by frequency and get top 5
                colors.sort(reverse=True)
                dominant_colors = []
                for count, color in colors[:5]:
                    if isinstance(color, tuple) and len(color) >= 3:
                        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                        dominant_colors.append(hex_color)
                analysis['dominant_colors'] = dominant_colors
            
            # Basic image statistics
            stat = ImageStat.Stat(img)
            if stat.mean:
                analysis['brightness_score'] = sum(stat.mean) / (len(stat.mean) * 255.0)
            if stat.stddev:
                analysis['contrast_score'] = sum(stat.stddev) / (len(stat.stddev) * 255.0)
            
            # Color distribution
            if len(img.getbands()) >= 3:
                r, g, b = img.split()[:3]
                r_hist = r.histogram()
                g_hist = g.histogram()
                b_hist = b.histogram()
                
                analysis['color_distribution'] = {
                    'red_intensity': sum(i * v for i, v in enumerate(r_hist)) / sum(r_hist) / 255.0,
                    'green_intensity': sum(i * v for i, v in enumerate(g_hist)) / sum(g_hist) / 255.0,
                    'blue_intensity': sum(i * v for i, v in enumerate(b_hist)) / sum(b_hist) / 255.0
                }
            
        except Exception as e:
            logger.warning(f"PIL analysis failed: {e}")
        
        return analysis
    
    async def _analyze_with_opencv(self, image_data: bytes) -> Dict[str, Any]:
        """Analyze image using OpenCV."""
        analysis = {}
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return analysis
            
            # Face detection (basic)
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Use default OpenCV face cascade if available
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                analysis['has_faces'] = len(faces) > 0
                analysis['face_count'] = len(faces)
                
            except Exception:
                # Face detection failed, probably due to missing cascade files
                analysis['has_faces'] = False
                analysis['face_count'] = 0
            
            # Text detection (very basic - look for text-like regions)
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to get binary image
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Look for rectangular contours that might be text
                text_like_contours = 0
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Text-like characteristics: certain aspect ratio, minimum size
                    if 0.2 < aspect_ratio < 10 and w > 10 and h > 5:
                        text_like_contours += 1
                
                analysis['has_text'] = text_like_contours > 5  # Threshold for text detection
                analysis['text_regions'] = text_like_contours
                
            except Exception:
                analysis['has_text'] = False
                analysis['text_regions'] = 0
            
            # Image quality metrics
            # Laplacian variance (sharpness)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            analysis['sharpness_score'] = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
        except Exception as e:
            logger.warning(f"OpenCV analysis failed: {e}")
        
        return analysis
    
    def _calculate_quality_score(self, analysis: Dict[str, Any], thumbnail: ThumbnailData) -> float:
        """Calculate overall quality score for thumbnail."""
        score = 0.0
        factors = 0
        
        # Resolution factor
        if thumbnail.width > 0 and thumbnail.height > 0:
            pixel_count = thumbnail.width * thumbnail.height
            if pixel_count >= 1280 * 720:  # HD
                score += 0.3
            elif pixel_count >= 640 * 480:  # SD
                score += 0.2
            elif pixel_count >= 320 * 180:  # Low
                score += 0.1
            factors += 1
        
        # Brightness factor
        brightness = analysis.get('brightness_score', 0)
        if 0.2 <= brightness <= 0.8:  # Good brightness range
            score += 0.2
        elif 0.1 <= brightness <= 0.9:  # Acceptable range
            score += 0.1
        factors += 1
        
        # Contrast factor
        contrast = analysis.get('contrast_score', 0)
        if contrast > 0.1:  # Good contrast
            score += 0.2
        elif contrast > 0.05:  # Some contrast
            score += 0.1
        factors += 1
        
        # Sharpness factor (if available)
        sharpness = analysis.get('sharpness_score', 0)
        if sharpness > 0.5:
            score += 0.2
        elif sharpness > 0.2:
            score += 0.1
        factors += 1
        
        # Color variety factor
        dominant_colors = analysis.get('dominant_colors', [])
        if len(dominant_colors) >= 3:
            score += 0.1
        factors += 1
        
        return score / factors if factors > 0 else 0.0


# Utility functions
def extract_youtube_thumbnail_urls(video_id: str) -> Dict[str, str]:
    """Generate all YouTube thumbnail URLs for a video ID."""
    base_url = f"https://img.youtube.com/vi/{video_id}"
    
    return {
        'default': f"{base_url}/default.jpg",
        'medium': f"{base_url}/mqdefault.jpg", 
        'high': f"{base_url}/hqdefault.jpg",
        'standard': f"{base_url}/sddefault.jpg",
        'maxres': f"{base_url}/maxresdefault.jpg"
    }


def compare_thumbnails(thumb1: ThumbnailData, thumb2: ThumbnailData) -> Dict[str, Any]:
    """Compare two thumbnails and return similarity metrics."""
    comparison = {
        'same_video': thumb1.video_id == thumb2.video_id,
        'size_difference': abs(thumb1.width * thumb1.height - thumb2.width * thumb2.height),
        'aspect_ratio_diff': abs(thumb1.get_aspect_ratio() - thumb2.get_aspect_ratio()),
        'quality_diff': abs(thumb1.quality_score - thumb2.quality_score),
        'similarity_score': 0.0
    }
    
    # Calculate simple similarity score
    score = 0.0
    if comparison['same_video']:
        score += 0.5
    
    if comparison['aspect_ratio_diff'] < 0.1:
        score += 0.2
    
    if comparison['quality_diff'] < 0.2:
        score += 0.2
    
    if thumb1.format == thumb2.format:
        score += 0.1
    
    comparison['similarity_score'] = score
    
    return comparison


def get_best_thumbnail(thumbnails: List[ThumbnailData]) -> Optional[ThumbnailData]:
    """Get the best thumbnail from a list based on quality metrics."""
    if not thumbnails:
        return None
    
    if len(thumbnails) == 1:
        return thumbnails[0]
    
    # Score each thumbnail
    scored_thumbnails = []
    for thumb in thumbnails:
        score = 0.0
        
        # Resolution score
        pixels = thumb.width * thumb.height
        if pixels >= 1280 * 720:
            score += 3
        elif pixels >= 640 * 480:
            score += 2
        elif pixels >= 320 * 180:
            score += 1
        
        # Quality score
        score += thumb.quality_score * 2
        
        # Format preference
        if thumb.format in ['jpg', 'jpeg']:
            score += 0.5
        elif thumb.format == 'png':
            score += 0.3
        
        # Size name preference
        size_preferences = {
            'maxres': 5,
            'high': 4,
            'standard': 3,
            'medium': 2,
            'default': 1
        }
        score += size_preferences.get(thumb.size_name, 0)
        
        scored_thumbnails.append((score, thumb))
    
    # Return thumbnail with highest score
    scored_thumbnails.sort(reverse=True)
    return scored_thumbnails[0][1]


__all__ = [
    'ThumbnailParser',
    'ThumbnailProcessor',
    'ThumbnailAnalyzer',
    'extract_youtube_thumbnail_urls',
    'compare_thumbnails',
    'get_best_thumbnail'
]