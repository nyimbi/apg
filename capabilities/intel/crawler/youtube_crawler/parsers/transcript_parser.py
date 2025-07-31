"""
Transcript Parser Module
========================

Specialized parsers for YouTube transcript and caption data extraction.
Handles VTT, SRT, and other subtitle formats with timing information.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import re
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

from .base_parser import BaseParser, ParseResult, ParseStatus, ContentType
from ..api.data_models import TranscriptData

logger = logging.getLogger(__name__)

# Optional dependency for language detection
try:
    from langdetect import detect, LangDetectError
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# Optional dependency for advanced text processing
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


class TranscriptParser(BaseParser):
    """Main transcript parser for various subtitle formats."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.subtitle_parser = SubtitleParser(**kwargs)
        self.caption_parser = CaptionParser(**kwargs)
        self.analyzer = TranscriptAnalyzer(**kwargs)
    
    async def parse(self, data: Union[Dict, str], content_type: ContentType) -> ParseResult:
        """Parse transcript data from various sources."""
        start_time = time.time()
        
        try:
            transcript_data = None
            
            if isinstance(data, str):
                # Raw transcript content (VTT, SRT, etc.)
                transcript_data = await self._parse_raw_transcript(data)
            elif isinstance(data, dict):
                if 'transcript' in data or 'text' in data:
                    # Structured transcript data
                    transcript_data = await self._parse_structured_transcript(data)
                elif 'items' in data:
                    # YouTube API transcript response
                    transcript_data = await self._parse_api_transcript(data)
                else:
                    # Single transcript item
                    transcript_data = await self._parse_transcript_item(data)
            
            if not transcript_data:
                raise ValueError("No valid transcript data found")
            
            # Analyze transcript if requested
            if self.config.analyze_content and transcript_data:
                analysis_result = await self.analyzer.analyze_transcript(transcript_data)
                if analysis_result.is_successful():
                    analysis_data = analysis_result.data
                    transcript_data.word_count = analysis_data.get('word_count', 0)
                    transcript_data.language_confidence = analysis_data.get('language_confidence', 0.0)
                    transcript_data.topics = analysis_data.get('topics', [])
                    transcript_data.keywords = analysis_data.get('keywords', [])
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=transcript_data,
                execution_time=execution_time,
                metadata={
                    'parser': 'TranscriptParser',
                    'format': transcript_data.format_type if transcript_data else 'unknown',
                    'language': transcript_data.language if transcript_data else 'unknown'
                }
            )
            
        except Exception as e:
            logger.error(f"Transcript parsing failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _parse_raw_transcript(self, content: str) -> Optional[TranscriptData]:
        """Parse raw transcript content (VTT, SRT, etc.)."""
        content = content.strip()
        
        # Detect format
        if content.startswith('WEBVTT'):
            return await self._parse_vtt_format(content)
        elif re.search(r'^\d+\s*$', content.split('\n')[0]):
            return await self._parse_srt_format(content)
        elif content.startswith('[') or content.startswith('{'):
            return await self._parse_json_format(content)
        else:
            # Plain text transcript
            return await self._parse_plain_text(content)
    
    async def _parse_vtt_format(self, content: str) -> TranscriptData:
        """Parse WebVTT format transcript."""
        lines = content.split('\n')
        text_parts = []
        segments = []
        total_duration = 0.0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and headers
            if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
                i += 1
                continue
            
            # Check for timestamp line
            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
            if timestamp_match:
                start_time = self._parse_vtt_timestamp(timestamp_match.group(1))
                end_time = self._parse_vtt_timestamp(timestamp_match.group(2))
                duration = end_time - start_time
                total_duration = max(total_duration, end_time)
                
                # Get subtitle text (next lines until empty line)
                i += 1
                subtitle_lines = []
                while i < len(lines) and lines[i].strip():
                    subtitle_text = lines[i].strip()
                    # Remove VTT formatting tags
                    subtitle_text = re.sub(r'<[^>]+>', '', subtitle_text)
                    subtitle_lines.append(subtitle_text)
                    i += 1
                
                if subtitle_lines:
                    segment_text = ' '.join(subtitle_lines)
                    text_parts.append(segment_text)
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration,
                        'text': segment_text
                    })
            else:
                i += 1
        
        full_text = ' '.join(text_parts)
        
        # Detect language
        language = 'en'  # Default
        if HAS_LANGDETECT and full_text:
            try:
                language = detect(full_text)
            except LangDetectError:
                pass
        
        return TranscriptData(
            video_id="unknown",  # Will be set by caller
            text=full_text,
            language=language,
            auto_generated=False,  # VTT is usually manual
            source_type="captions",
            format_type="vtt",
            start_time=0.0,
            duration=total_duration,
            segments=segments,
            word_count=len(full_text.split()) if full_text else 0
        )
    
    async def _parse_srt_format(self, content: str) -> TranscriptData:
        """Parse SRT format transcript."""
        lines = content.split('\n')
        text_parts = []
        segments = []
        total_duration = 0.0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Check for sequence number
            if line.isdigit():
                i += 1
                if i >= len(lines):
                    break
                
                # Check for timestamp line
                timestamp_line = lines[i].strip()
                timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
                if timestamp_match:
                    start_time = self._parse_srt_timestamp(timestamp_match.group(1))
                    end_time = self._parse_srt_timestamp(timestamp_match.group(2))
                    duration = end_time - start_time
                    total_duration = max(total_duration, end_time)
                    
                    # Get subtitle text
                    i += 1
                    subtitle_lines = []
                    while i < len(lines) and lines[i].strip():
                        subtitle_lines.append(lines[i].strip())
                        i += 1
                    
                    if subtitle_lines:
                        segment_text = ' '.join(subtitle_lines)
                        text_parts.append(segment_text)
                        segments.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration,
                            'text': segment_text
                        })
            else:
                i += 1
        
        full_text = ' '.join(text_parts)
        
        # Detect language
        language = 'en'  # Default
        if HAS_LANGDETECT and full_text:
            try:
                language = detect(full_text)
            except LangDetectError:
                pass
        
        return TranscriptData(
            video_id="unknown",
            text=full_text,
            language=language,
            auto_generated=False,
            source_type="subtitles",
            format_type="srt",
            start_time=0.0,
            duration=total_duration,
            segments=segments,
            word_count=len(full_text.split()) if full_text else 0
        )
    
    async def _parse_json_format(self, content: str) -> TranscriptData:
        """Parse JSON format transcript."""
        try:
            data = json.loads(content)
            
            if isinstance(data, list):
                # Array of transcript segments
                segments = []
                text_parts = []
                total_duration = 0.0
                
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                        
                        segment = {
                            'text': item['text'],
                            'start_time': float(item.get('start', 0)),
                            'duration': float(item.get('dur', 0))
                        }
                        segment['end_time'] = segment['start_time'] + segment['duration']
                        segments.append(segment)
                        total_duration = max(total_duration, segment['end_time'])
                
                full_text = ' '.join(text_parts)
                
                return TranscriptData(
                    video_id="unknown",
                    text=full_text,
                    language=data[0].get('lang', 'en') if data else 'en',
                    auto_generated=True,  # JSON format usually from auto-generated
                    source_type="captions",
                    format_type="json",
                    start_time=0.0,
                    duration=total_duration,
                    segments=segments,
                    word_count=len(full_text.split()) if full_text else 0
                )
            
            elif isinstance(data, dict):
                # Single transcript object
                text = data.get('text', '')
                return TranscriptData(
                    video_id=data.get('video_id', "unknown"),
                    text=text,
                    language=data.get('language', 'en'),
                    auto_generated=data.get('auto_generated', True),
                    source_type=data.get('source_type', 'captions'),
                    format_type="json",
                    start_time=float(data.get('start_time', 0)),
                    duration=float(data.get('duration', 0)),
                    segments=data.get('segments', []),
                    word_count=len(text.split()) if text else 0
                )
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON transcript format: {e}")
            return None
    
    async def _parse_plain_text(self, content: str) -> TranscriptData:
        """Parse plain text transcript."""
        text = content.strip()
        
        # Detect language
        language = 'en'
        if HAS_LANGDETECT and text:
            try:
                language = detect(text)
            except LangDetectError:
                pass
        
        return TranscriptData(
            video_id="unknown",
            text=text,
            language=language,
            auto_generated=False,
            source_type="manual",
            format_type="text",
            start_time=0.0,
            duration=0.0,
            segments=[],
            word_count=len(text.split()) if text else 0
        )
    
    async def _parse_structured_transcript(self, data: Dict[str, Any]) -> TranscriptData:
        """Parse structured transcript data."""
        text = data.get('transcript', data.get('text', ''))
        
        return TranscriptData(
            video_id=data.get('video_id', "unknown"),
            text=text,
            language=data.get('language', 'en'),
            auto_generated=data.get('auto_generated', False),
            source_type=data.get('source_type', 'captions'),
            format_type=data.get('format_type', 'unknown'),
            start_time=float(data.get('start_time', 0)),
            duration=float(data.get('duration', 0)),
            confidence=data.get('confidence'),
            segments=data.get('segments', []),
            word_count=len(text.split()) if text else 0
        )
    
    async def _parse_api_transcript(self, data: Dict[str, Any]) -> TranscriptData:
        """Parse YouTube API transcript response."""
        items = data.get('items', [])
        text_parts = []
        segments = []
        total_duration = 0.0
        
        for item in items:
            snippet = item.get('snippet', {})
            text_parts.append(snippet.get('text', ''))
            
            start_time = float(snippet.get('start', 0))
            duration = float(snippet.get('dur', 0))
            end_time = start_time + duration
            total_duration = max(total_duration, end_time)
            
            segments.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'text': snippet.get('text', '')
            })
        
        full_text = ' '.join(text_parts)
        
        return TranscriptData(
            video_id=data.get('video_id', "unknown"),
            text=full_text,
            language=data.get('language', 'en'),
            auto_generated=True,
            source_type="captions",
            format_type="api",
            start_time=0.0,
            duration=total_duration,
            segments=segments,
            word_count=len(full_text.split()) if full_text else 0
        )
    
    async def _parse_transcript_item(self, data: Dict[str, Any]) -> TranscriptData:
        """Parse single transcript item."""
        return await self._parse_structured_transcript(data)
    
    def _parse_vtt_timestamp(self, timestamp: str) -> float:
        """Parse VTT timestamp (HH:MM:SS.mmm) to seconds."""
        parts = timestamp.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1])
        
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    
    def _parse_srt_timestamp(self, timestamp: str) -> float:
        """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
        parts = timestamp.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split(',')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1])
        
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


class SubtitleParser(BaseParser):
    """Specialized parser for subtitle files."""
    
    async def parse(self, data: Union[str, Dict], content_type: ContentType) -> ParseResult:
        """Parse subtitle data."""
        # This is handled by the main TranscriptParser
        # Keeping as a separate class for modularity
        transcript_parser = TranscriptParser()
        return await transcript_parser.parse(data, content_type)


class CaptionParser(BaseParser):
    """Specialized parser for closed captions."""
    
    async def parse(self, data: Union[str, Dict], content_type: ContentType) -> ParseResult:
        """Parse caption data."""
        # This is handled by the main TranscriptParser
        # Keeping as a separate class for modularity
        transcript_parser = TranscriptParser()
        return await transcript_parser.parse(data, content_type)


class TranscriptAnalyzer(BaseParser):
    """Analyzer for transcript content analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stopwords = set()
        
        if HAS_NLTK:
            try:
                self.stopwords = set(stopwords.words('english'))
            except LookupError:
                logger.warning("NLTK stopwords not found. Run nltk.download('stopwords')")
    
    async def analyze_transcript(self, transcript: TranscriptData) -> ParseResult:
        """Analyze transcript content."""
        start_time = time.time()
        
        try:
            analysis = {
                'word_count': 0,
                'unique_words': 0,
                'language_confidence': 0.0,
                'topics': [],
                'keywords': [],
                'reading_time_minutes': 0.0,
                'speaking_rate_wpm': 0.0,
                'text_complexity': 0.0
            }
            
            if not transcript.text:
                return ParseResult(
                    status=ParseStatus.SUCCESS,
                    data=analysis,
                    execution_time=time.time() - start_time
                )
            
            # Word count analysis
            words = transcript.text.split()
            analysis['word_count'] = len(words)
            analysis['unique_words'] = len(set(word.lower() for word in words))
            
            # Reading time (average 200 WPM)
            analysis['reading_time_minutes'] = len(words) / 200.0
            
            # Speaking rate
            if transcript.duration > 0:
                analysis['speaking_rate_wpm'] = (len(words) / transcript.duration) * 60.0
            
            # Language confidence
            if HAS_LANGDETECT:
                try:
                    detected_lang = detect(transcript.text)
                    analysis['language_confidence'] = 0.9 if detected_lang == transcript.language else 0.5
                except LangDetectError:
                    analysis['language_confidence'] = 0.3
            
            # Extract keywords
            keywords = await self._extract_keywords(transcript.text)
            analysis['keywords'] = keywords[:20]  # Top 20 keywords
            
            # Extract topics (simple approach)
            topics = await self._extract_topics(transcript.text)
            analysis['topics'] = topics
            
            # Text complexity (simple metric)
            analysis['text_complexity'] = self._calculate_text_complexity(transcript.text)
            
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.SUCCESS,
                data=analysis,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Transcript analysis failed: {e}")
            execution_time = time.time() - start_time
            
            return ParseResult(
                status=ParseStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from transcript text."""
        if HAS_NLTK:
            try:
                # Tokenize and filter
                words = word_tokenize(text.lower())
                words = [word for word in words if word.isalnum() and len(word) > 3]
                words = [word for word in words if word not in self.stopwords]
                
                # Get frequency distribution
                from collections import Counter
                word_freq = Counter(words)
                
                return [word for word, freq in word_freq.most_common(50) if freq > 1]
            except Exception as e:
                logger.warning(f"NLTK keyword extraction failed: {e}")
        
        # Fallback: simple word frequency
        words = text.lower().split()
        words = [word.strip('.,!?;:"()[]{}') for word in words]
        words = [word for word in words if len(word) > 3 and word.isalnum()]
        
        from collections import Counter
        word_freq = Counter(words)
        
        return [word for word, freq in word_freq.most_common(30) if freq > 1]
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from transcript text (simple approach)."""
        # Simple topic extraction based on common topic keywords
        topic_keywords = {
            'technology': ['technology', 'software', 'computer', 'digital', 'internet', 'ai', 'artificial intelligence'],
            'science': ['science', 'research', 'study', 'experiment', 'theory', 'hypothesis'],
            'education': ['education', 'learning', 'school', 'university', 'student', 'teacher'],
            'business': ['business', 'company', 'market', 'economy', 'finance', 'investment'],
            'health': ['health', 'medical', 'doctor', 'patient', 'treatment', 'medicine'],
            'politics': ['politics', 'government', 'election', 'policy', 'law', 'political'],
            'entertainment': ['movie', 'music', 'game', 'entertainment', 'celebrity', 'show'],
            'sports': ['sports', 'football', 'basketball', 'soccer', 'athlete', 'competition'],
            'travel': ['travel', 'trip', 'vacation', 'country', 'city', 'tourism'],
            'food': ['food', 'cooking', 'recipe', 'restaurant', 'chef', 'cuisine']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:  # Require at least 2 keyword matches
                detected_topics.append(topic)
        
        return detected_topics
    
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score (0-1)."""
        if not text:
            return 0.0
        
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Average syllables per word (approximation)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Flesch Reading Ease approximation
        flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables)
        
        # Convert to 0-1 scale (higher = more complex)
        complexity = max(0, min(1, (100 - flesch_score) / 100))
        
        return complexity
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_vowel:
                    syllable_count += 1
                prev_char_vowel = True
            else:
                prev_char_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)  # At least 1 syllable


# Utility functions
def extract_time_segments(transcript: TranscriptData, duration_threshold: float = 30.0) -> List[Dict[str, Any]]:
    """Extract time segments from transcript."""
    if not transcript.segments:
        return []
    
    segments = []
    current_segment = {
        'start_time': 0,
        'end_time': 0,
        'text': '',
        'word_count': 0
    }
    
    for segment in transcript.segments:
        segment_duration = segment['end_time'] - current_segment['start_time']
        
        if segment_duration > duration_threshold and current_segment['text']:
            # Close current segment and start new one
            segments.append(current_segment.copy())
            current_segment = {
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'text': segment['text'],
                'word_count': len(segment['text'].split())
            }
        else:
            # Continue current segment
            if not current_segment['text']:
                current_segment['start_time'] = segment['start_time']
            current_segment['end_time'] = segment['end_time']
            current_segment['text'] += ' ' + segment['text']
            current_segment['word_count'] += len(segment['text'].split())
    
    # Add final segment
    if current_segment['text']:
        segments.append(current_segment)
    
    return segments


def search_transcript(transcript: TranscriptData, query: str, context_words: int = 10) -> List[Dict[str, Any]]:
    """Search for query in transcript with context."""
    if not transcript.text or not query:
        return []
    
    words = transcript.text.split()
    query_words = query.lower().split()
    results = []
    
    for i, word in enumerate(words):
        if word.lower() in [q.lower() for q in query_words]:
            # Found a match, get context
            start_idx = max(0, i - context_words)
            end_idx = min(len(words), i + context_words + 1)
            
            context = ' '.join(words[start_idx:end_idx])
            
            # Find corresponding time segment if available
            time_info = None
            if transcript.segments:
                word_position = len(' '.join(words[:i]))
                for segment in transcript.segments:
                    if segment['text'] in context:
                        time_info = {
                            'start_time': segment['start_time'],
                            'end_time': segment['end_time']
                        }
                        break
            
            results.append({
                'match_word': word,
                'context': context,
                'word_position': i,
                'time_info': time_info
            })
    
    return results


def get_transcript_summary(transcript: TranscriptData) -> Dict[str, Any]:
    """Get summary statistics for transcript."""
    summary = {
        'video_id': transcript.video_id,
        'language': transcript.language,
        'auto_generated': transcript.auto_generated,
        'format_type': transcript.format_type,
        'duration_minutes': transcript.duration / 60.0 if transcript.duration else 0,
        'word_count': transcript.word_count,
        'segment_count': len(transcript.segments),
        'has_timing': len(transcript.segments) > 0,
        'speaking_rate_wpm': transcript.get_words_per_minute() if hasattr(transcript, 'get_words_per_minute') else 0
    }
    
    if transcript.confidence:
        summary['confidence'] = transcript.confidence
    
    if transcript.topics:
        summary['topics'] = transcript.topics
    
    if transcript.keywords:
        summary['top_keywords'] = transcript.keywords[:10]
    
    return summary


__all__ = [
    'TranscriptParser',
    'SubtitleParser', 
    'CaptionParser',
    'TranscriptAnalyzer',
    'extract_time_segments',
    'search_transcript',
    'get_transcript_summary'
]