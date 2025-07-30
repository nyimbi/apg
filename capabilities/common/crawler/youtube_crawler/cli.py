"""
Comprehensive YouTube Crawler CLI
=================================

Advanced command-line interface for YouTube content crawling, analysis, and downloading.
Supports multiple operation modes, configuration management, and youtube-dl integration.

Features:
- Search and analyze YouTube content
- Download videos and extract audio
- Monitor channels and playlists
- Batch processing with queue management
- Interactive mode for exploratory analysis
- Configuration management with YAML/JSON support
- Performance monitoring and health checks
- Export to multiple formats (JSON, CSV, Excel)

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import click
import json
import yaml
import logging
import sys
import time
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
from dataclasses import asdict
import signal

# Rich imports for enhanced CLI output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.layout import Layout
    from rich.live import Live
    from rich import print as rich_print
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

# Tabulate for simple table output
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Pandas for data export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Local imports
try:
    from .optimized_core import OptimizedYouTubeCrawler, OptimizedYouTubeConfig, create_optimized_youtube_crawler
    from .api.data_models import VideoData, ChannelData
    from .api.exceptions import YouTubeCrawlerError
    OPTIMIZED_CORE_AVAILABLE = True
except ImportError:
    OPTIMIZED_CORE_AVAILABLE = False
    logging.warning("Optimized core not available, using fallback")

console = Console() if RICH_AVAILABLE else None


class OperationMode(Enum):
    """CLI operation modes."""
    SEARCH = "search"
    DOWNLOAD = "download"
    ANALYZE = "analyze"
    MONITOR = "monitor"
    BATCH = "batch"
    INTERACTIVE = "interactive"
    HEALTH = "health"


class OutputFormat(Enum):
    """Output format options."""
    TABLE = "table"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    YAML = "yaml"


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-essential output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Set logging level')
@click.pass_context
def main(ctx, verbose, quiet, config, log_level):
    """
    🎬 Enhanced YouTube Crawler CLI
    
    Advanced YouTube content analysis, search, and download tool with performance optimizations.
    """
    # Setup logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if verbose:
        log_level = 'DEBUG'
    elif quiet:
        log_level = 'WARNING'
    
    logging.basicConfig(level=getattr(logging, log_level), format=log_format)
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    ctx.obj['config_file'] = config
    ctx.obj['log_level'] = log_level
    
    # Load configuration
    ctx.obj['config'] = load_configuration(config)


def load_configuration(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or create default."""
    default_config = {
        'youtube_api_key': os.getenv('YOUTUBE_API_KEY'),
        'rate_limit_requests_per_minute': 60,
        'max_concurrent_requests': 20,
        'max_concurrent_downloads': 5,
        'enable_video_download': False,
        'download_format': 'best[height<=720]',
        'download_audio_only': False,
        'max_download_size_mb': 500,
        'output_directory': './downloads',
        'cache_enabled': True,
        'cache_ttl': 3600,
        'enable_performance_monitoring': True,
        'health_check_interval': 300
    }
    
    if not config_path:
        return default_config
    
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # Merge with defaults
        default_config.update(config_data)
        return default_config
        
    except Exception as e:
        if console:
            console.print(f"[red]Error loading config: {e}[/red]")
        else:
            print(f"Error loading config: {e}")
        return default_config


@main.command()
@click.argument('query')
@click.option('--max-results', '-n', default=50, help='Maximum number of results')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json', 'csv', 'excel']), 
              default='table', help='Output format')
@click.option('--output', '-o', help='Output file path')
@click.option('--save-details', is_flag=True, help='Save detailed video information')
@click.option('--download', is_flag=True, help='Download videos after search')
@click.option('--audio-only', is_flag=True, help='Download audio only')
@click.option('--language', default='en', help='Search language')
@click.option('--region', default='US', help='Search region')
@click.option('--published-after', help='Videos published after date (YYYY-MM-DD)')
@click.option('--published-before', help='Videos published before date (YYYY-MM-DD)')
@click.option('--duration', type=click.Choice(['short', 'medium', 'long']), help='Video duration filter')
@click.option('--order', type=click.Choice(['relevance', 'date', 'rating', 'title', 'viewCount']), 
              default='relevance', help='Sort order')
@click.pass_context
async def search(ctx, query, max_results, output_format, output, save_details, download, 
                audio_only, language, region, published_after, published_before, duration, order):
    """
    🔍 Search for YouTube videos with advanced filtering.
    
    QUERY: Search query string
    """
    config_data = ctx.obj['config']
    
    if not OPTIMIZED_CORE_AVAILABLE:
        if console:
            console.print("[red]Error: Optimized core not available. Please install dependencies.[/red]")
        else:
            print("Error: Optimized core not available. Please install dependencies.")
        sys.exit(1)
    
    try:
        # Create crawler
        crawler_config = OptimizedYouTubeConfig(
            youtube_api_key=config_data.get('youtube_api_key'),
            rate_limit_requests_per_minute=config_data.get('rate_limit_requests_per_minute', 60),
            max_concurrent_requests=config_data.get('max_concurrent_requests', 20),
            enable_video_download=download,
            download_audio_only=audio_only,
            enable_performance_monitoring=config_data.get('enable_performance_monitoring', True)
        )
        
        crawler = await create_optimized_youtube_crawler(crawler_config)
        
        # Build search parameters
        search_params = {
            'regionCode': region,
            'relevanceLanguage': language,
            'order': order
        }
        
        if published_after:
            search_params['publishedAfter'] = f"{published_after}T00:00:00Z"
        if published_before:
            search_params['publishedBefore'] = f"{published_before}T23:59:59Z"
        if duration:
            search_params['videoDuration'] = duration
        
        # Show progress
        if console and not ctx.obj['quiet']:
            with console.status(f"[bold green]Searching for: {query}"):
                videos = await crawler.search_videos(query, max_results, **search_params)
        else:
            videos = await crawler.search_videos(query, max_results, **search_params)
        
        # Process results
        if save_details:
            if console and not ctx.obj['quiet']:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Fetching details...", total=len(videos))
                    
                    detailed_videos = []
                    for video in videos:
                        try:
                            detailed_video = await crawler.get_video_details(video.video_id)
                            detailed_videos.append(detailed_video)
                        except Exception as e:
                            logging.warning(f"Failed to get details for {video.video_id}: {e}")
                            detailed_videos.append(video)
                        progress.advance(task)
                    
                    videos = detailed_videos
            else:
                detailed_videos = []
                for video in videos:
                    try:
                        detailed_video = await crawler.get_video_details(video.video_id)
                        detailed_videos.append(detailed_video)
                    except Exception:
                        detailed_videos.append(video)
                videos = detailed_videos
        
        # Download videos if requested
        if download:
            download_results = []
            output_dir = config_data.get('output_directory', './downloads')
            
            if console and not ctx.obj['quiet']:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Downloading videos...", total=len(videos))
                    
                    for video in videos:
                        try:
                            if audio_only:
                                result = await crawler.extract_audio(video.video_id, output_dir)
                            else:
                                result = await crawler.download_video(video.video_id, output_dir)
                            download_results.append(result)
                        except Exception as e:
                            logging.error(f"Download failed for {video.video_id}: {e}")
                            download_results.append({
                                'success': False,
                                'video_id': video.video_id,
                                'error': str(e)
                            })
                        progress.advance(task)
            else:
                for video in videos:
                    try:
                        if audio_only:
                            result = await crawler.extract_audio(video.video_id, output_dir)
                        else:
                            result = await crawler.download_video(video.video_id, output_dir)
                        download_results.append(result)
                    except Exception as e:
                        logging.error(f"Download failed for {video.video_id}: {e}")
                        download_results.append({
                            'success': False,
                            'video_id': video.video_id,
                            'error': str(e)
                        })
            
            # Show download summary
            successful_downloads = sum(1 for r in download_results if r.get('success', False))
            if console:
                console.print(f"\n[green]Downloads completed: {successful_downloads}/{len(download_results)}[/green]")
            else:
                print(f"\nDownloads completed: {successful_downloads}/{len(download_results)}")
        
        # Output results
        await output_results(videos, output_format, output, ctx.obj['quiet'])
        
        # Show summary
        if not ctx.obj['quiet']:
            if console:
                console.print(f"\n[green]Found {len(videos)} videos for query: {query}[/green]")
            else:
                print(f"\nFound {len(videos)} videos for query: {query}")
        
        await crawler.close()
        
    except Exception as e:
        if console:
            console.print(f"[red]Search failed: {e}[/red]")
        else:
            print(f"Search failed: {e}")
        sys.exit(1)


@main.command()
@click.argument('video_ids', nargs=-1)
@click.option('--format', 'download_format', default='best[height<=720]', 
              help='Download format (yt-dlp format string)')
@click.option('--audio-only', is_flag=True, help='Download audio only')
@click.option('--output-dir', '-o', default='./downloads', help='Output directory')
@click.option('--max-size', default=500, help='Maximum file size in MB')
@click.option('--subtitles', is_flag=True, help='Download subtitles')
@click.option('--thumbnails', is_flag=True, help='Download thumbnails')
@click.option('--metadata', is_flag=True, help='Save metadata to file')
@click.option('--concurrent', default=5, help='Number of concurrent downloads')
@click.option('--playlist', help='Download from playlist URL')
@click.option('--channel', help='Download from channel URL')
@click.option('--input-file', help='Read video IDs from file')
@click.pass_context
async def download(ctx, video_ids, download_format, audio_only, output_dir, max_size, 
                  subtitles, thumbnails, metadata, concurrent, playlist, channel, input_file):
    """
    ⬇️ Download YouTube videos with advanced options.
    
    VIDEO_IDS: One or more YouTube video IDs or URLs
    """
    config_data = ctx.obj['config']
    
    if not OPTIMIZED_CORE_AVAILABLE:
        if console:
            console.print("[red]Error: Optimized core not available. Please install dependencies.[/red]")
        else:
            print("Error: Optimized core not available. Please install dependencies.")
        sys.exit(1)
    
    # Collect video IDs from various sources
    all_video_ids = list(video_ids)
    
    # Read from input file
    if input_file:
        try:
            with open(input_file, 'r') as f:
                file_ids = [line.strip() for line in f if line.strip()]
                all_video_ids.extend(file_ids)
        except Exception as e:
            if console:
                console.print(f"[red]Error reading input file: {e}[/red]")
            else:
                print(f"Error reading input file: {e}")
            sys.exit(1)
    
    # Extract video IDs from URLs
    extracted_ids = []
    for vid_id in all_video_ids:
        if 'youtube.com/watch?v=' in vid_id:
            extracted_ids.append(vid_id.split('v=')[1].split('&')[0])
        elif 'youtu.be/' in vid_id:
            extracted_ids.append(vid_id.split('/')[-1].split('?')[0])
        else:
            extracted_ids.append(vid_id)
    
    all_video_ids = extracted_ids
    
    if not all_video_ids and not playlist and not channel:
        if console:
            console.print("[red]Error: No video IDs provided[/red]")
        else:
            print("Error: No video IDs provided")
        sys.exit(1)
    
    try:
        # Create crawler
        crawler_config = OptimizedYouTubeConfig(
            enable_video_download=True,
            download_format=download_format,
            download_audio_only=audio_only,
            download_subtitles=subtitles,
            download_thumbnails=thumbnails,
            max_download_size_mb=max_size,
            max_concurrent_downloads=concurrent
        )
        
        crawler = await create_optimized_youtube_crawler(crawler_config)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Download videos
        download_results = []
        
        if console and not ctx.obj['quiet']:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Downloading videos...", total=len(all_video_ids))
                
                for video_id in all_video_ids:
                    try:
                        if audio_only:
                            result = await crawler.extract_audio(video_id, output_dir)
                        else:
                            result = await crawler.download_video(video_id, output_dir)
                        
                        download_results.append(result)
                        
                        if metadata and result.get('success'):
                            # Save metadata
                            metadata_file = Path(output_dir) / f"{video_id}_metadata.json"
                            with open(metadata_file, 'w') as f:
                                json.dump(result, f, indent=2, default=str)
                        
                    except Exception as e:
                        logging.error(f"Download failed for {video_id}: {e}")
                        download_results.append({
                            'success': False,
                            'video_id': video_id,
                            'error': str(e)
                        })
                    
                    progress.advance(task)
        else:
            for video_id in all_video_ids:
                try:
                    if audio_only:
                        result = await crawler.extract_audio(video_id, output_dir)
                    else:
                        result = await crawler.download_video(video_id, output_dir)
                    
                    download_results.append(result)
                    
                    if metadata and result.get('success'):
                        metadata_file = Path(output_dir) / f"{video_id}_metadata.json"
                        with open(metadata_file, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                
                except Exception as e:
                    logging.error(f"Download failed for {video_id}: {e}")
                    download_results.append({
                        'success': False,
                        'video_id': video_id,
                        'error': str(e)
                    })
        
        # Show results
        successful = sum(1 for r in download_results if r.get('success', False))
        failed = len(download_results) - successful
        
        if console:
            console.print(f"\n[green]✅ Downloads completed: {successful}[/green]")
            if failed > 0:
                console.print(f"[red]❌ Failed downloads: {failed}[/red]")
        else:
            print(f"\nDownloads completed: {successful}")
            if failed > 0:
                print(f"Failed downloads: {failed}")
        
        # Save download summary
        summary_file = Path(output_dir) / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_requested': len(all_video_ids),
                'successful': successful,
                'failed': failed,
                'results': download_results
            }, f, indent=2, default=str)
        
        await crawler.close()
        
    except Exception as e:
        if console:
            console.print(f"[red]Download failed: {e}[/red]")
        else:
            print(f"Download failed: {e}")
        sys.exit(1)


@main.command()
@click.pass_context
async def interactive(ctx):
    """
    🎯 Interactive YouTube analysis mode.
    """
    if not RICH_AVAILABLE:
        print("Interactive mode requires 'rich' library. Please install: pip install rich")
        sys.exit(1)
    
    config_data = ctx.obj['config']
    
    if not OPTIMIZED_CORE_AVAILABLE:
        console.print("[red]Error: Optimized core not available. Please install dependencies.[/red]")
        sys.exit(1)
    
    # Create crawler
    crawler_config = OptimizedYouTubeConfig(
        youtube_api_key=config_data.get('youtube_api_key'),
        enable_performance_monitoring=True
    )
    
    crawler = await create_optimized_youtube_crawler(crawler_config)
    
    console.print(Panel.fit(
        "[bold blue]🎬 YouTube Crawler Interactive Mode[/bold blue]\n\n"
        "Available commands:\n"
        "• [green]search[/green] - Search for videos\n"
        "• [green]download[/green] - Download video by ID\n"
        "• [green]analyze[/green] - Analyze video details\n"
        "• [green]health[/green] - Show health status\n"
        "• [green]stats[/green] - Show performance stats\n"
        "• [green]config[/green] - Show configuration\n"
        "• [green]help[/green] - Show this help\n"
        "• [green]quit[/green] - Exit interactive mode",
        title="Welcome"
    ))
    
    try:
        while True:
            command = Prompt.ask("\n[bold cyan]youtube-crawler[/bold cyan]", default="help")
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.lower() == 'help':
                console.print(Panel.fit(
                    "[green]search <query>[/green] - Search videos\n"
                    "[green]download <video_id>[/green] - Download video\n"
                    "[green]analyze <video_id>[/green] - Analyze video\n"
                    "[green]health[/green] - Health check\n"
                    "[green]stats[/green] - Performance stats\n"
                    "[green]config[/green] - Show config\n"
                    "[green]quit[/green] - Exit",
                    title="Commands"
                ))
            elif command.lower().startswith('search '):
                query = command[7:].strip()
                if query:
                    await interactive_search(crawler, query, console)
                else:
                    console.print("[red]Please provide a search query[/red]")
            elif command.lower().startswith('download '):
                video_id = command[9:].strip()
                if video_id:
                    await interactive_download(crawler, video_id, console)
                else:
                    console.print("[red]Please provide a video ID[/red]")
            elif command.lower().startswith('analyze '):
                video_id = command[8:].strip()
                if video_id:
                    await interactive_analyze(crawler, video_id, console)
                else:
                    console.print("[red]Please provide a video ID[/red]")
            elif command.lower() == 'health':
                await interactive_health(crawler, console)
            elif command.lower() == 'stats':
                await interactive_stats(crawler, console)
            elif command.lower() == 'config':
                await interactive_config(config_data, console)
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type 'help' for available commands")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    finally:
        await crawler.close()
        console.print("[green]👋 Goodbye![/green]")


async def interactive_search(crawler, query, console):
    """Interactive search function."""
    try:
        with console.status(f"[bold green]Searching for: {query}"):
            videos = await crawler.search_videos(query, max_results=10)
        
        if videos:
            table = Table(title=f"Search Results for: {query}")
            table.add_column("ID", style="cyan")
            table.add_column("Title", style="green")
            table.add_column("Channel", style="yellow")
            table.add_column("Views", style="magenta")
            table.add_column("Duration", style="blue")
            
            for video in videos[:10]:
                table.add_row(
                    video.video_id[:11],
                    video.title[:50] + "..." if len(video.title) > 50 else video.title,
                    video.channel_name[:20] + "..." if len(video.channel_name) > 20 else video.channel_name,
                    str(video.view_count) if video.view_count else "N/A",
                    str(video.duration) if video.duration else "N/A"
                )
            
            console.print(table)
        else:
            console.print("[red]No videos found[/red]")
    
    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")


async def interactive_download(crawler, video_id, console):
    """Interactive download function."""
    try:
        with console.status(f"[bold green]Downloading: {video_id}"):
            result = await crawler.download_video(video_id, "./downloads")
        
        if result.get('success'):
            console.print(f"[green]✅ Download successful: {result.get('title', video_id)}[/green]")
        else:
            console.print(f"[red]❌ Download failed: {result.get('error', 'Unknown error')}[/red]")
    
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")


async def interactive_analyze(crawler, video_id, console):
    """Interactive analysis function."""
    try:
        with console.status(f"[bold green]Analyzing: {video_id}"):
            video = await crawler.get_video_details(video_id)
        
        # Create analysis panel
        analysis_text = f"""
[bold]Title:[/bold] {video.title}
[bold]Channel:[/bold] {video.channel_name}
[bold]Views:[/bold] {video.view_count:,} views
[bold]Likes:[/bold] {video.like_count:,} likes
[bold]Upload Date:[/bold] {video.upload_date}
[bold]Description:[/bold] {video.description[:200]}...
[bold]URL:[/bold] {video.video_url}
        """
        
        console.print(Panel(analysis_text, title=f"Video Analysis: {video_id}"))
    
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")


async def interactive_health(crawler, console):
    """Interactive health check function."""
    try:
        health_data = await crawler.health_check()
        
        status_color = "green" if health_data['is_healthy'] else "red"
        status_text = "Healthy" if health_data['is_healthy'] else "Unhealthy"
        
        health_text = f"""
[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]
[bold]Uptime:[/bold] {health_data['uptime']:.1f} seconds
[bold]Consecutive Failures:[/bold] {health_data['consecutive_failures']}
[bold]Memory Cache Size:[/bold] {health_data['cache_stats']['memory_cache_size']}
[bold]Active Connections:[/bold] {health_data['connection_pool_stats']['active_connections']}
        """
        
        console.print(Panel(health_text, title="Health Status"))
    
    except Exception as e:
        console.print(f"[red]Health check failed: {e}[/red]")


async def interactive_stats(crawler, console):
    """Interactive stats display function."""
    try:
        health_data = await crawler.health_check()
        stats = health_data.get('performance_stats', {})
        
        if stats:
            stats_text = f"""
[bold]Requests:[/bold] {stats.get('request_count', 0)}
[bold]Success Rate:[/bold] {stats.get('success_rate', 0):.2%}
[bold]Avg Response Time:[/bold] {stats.get('avg_response_time', 0):.2f}s
[bold]Requests/Second:[/bold] {stats.get('requests_per_second', 0):.2f}
[bold]Memory Usage:[/bold] {stats.get('current_memory_mb', 0):.1f} MB
[bold]Peak Memory:[/bold] {stats.get('peak_memory_mb', 0):.1f} MB
            """
            
            console.print(Panel(stats_text, title="Performance Statistics"))
        else:
            console.print("[yellow]No performance stats available[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Stats retrieval failed: {e}[/red]")


async def interactive_config(config_data, console):
    """Interactive config display function."""
    config_text = ""
    for key, value in config_data.items():
        if key.endswith('_key') and value:
            value = "*" * len(str(value))  # Hide API keys
        config_text += f"[bold]{key}:[/bold] {value}\n"
    
    console.print(Panel(config_text, title="Configuration"))


@main.command()
@click.pass_context
async def health(ctx):
    """
    🏥 Perform health check on YouTube crawler.
    """
    config_data = ctx.obj['config']
    
    if not OPTIMIZED_CORE_AVAILABLE:
        if console:
            console.print("[red]Error: Optimized core not available.[/red]")
        else:
            print("Error: Optimized core not available.")
        sys.exit(1)
    
    try:
        crawler_config = OptimizedYouTubeConfig(
            youtube_api_key=config_data.get('youtube_api_key'),
            enable_performance_monitoring=True
        )
        
        crawler = await create_optimized_youtube_crawler(crawler_config)
        health_data = await crawler.health_check()
        
        if ctx.obj['verbose'] or ctx.obj.get('format') == 'json':
            print(json.dumps(health_data, indent=2, default=str))
        else:
            status = "Healthy" if health_data['is_healthy'] else "Unhealthy"
            if console:
                color = "green" if health_data['is_healthy'] else "red"
                console.print(f"[{color}]Status: {status}[/{color}]")
                console.print(f"Uptime: {health_data['uptime']:.1f} seconds")
                console.print(f"Consecutive failures: {health_data['consecutive_failures']}")
            else:
                print(f"Status: {status}")
                print(f"Uptime: {health_data['uptime']:.1f} seconds")
                print(f"Consecutive failures: {health_data['consecutive_failures']}")
        
        await crawler.close()
        
    except Exception as e:
        if console:
            console.print(f"[red]Health check failed: {e}[/red]")
        else:
            print(f"Health check failed: {e}")
        sys.exit(1)


async def output_results(data: List[Any], format_type: str, output_path: Optional[str], quiet: bool):
    """Output results in specified format."""
    if not data:
        if not quiet:
            if console:
                console.print("[yellow]No data to output[/yellow]")
            else:
                print("No data to output")
        return
    
    # Convert data to dictionaries
    if hasattr(data[0], '__dict__'):
        data_dicts = [asdict(item) if hasattr(item, '__dict__') else item.__dict__ for item in data]
    else:
        data_dicts = data
    
    if format_type == 'json':
        output_data = json.dumps(data_dicts, indent=2, default=str)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(output_data)
        else:
            print(output_data)
    
    elif format_type == 'table':
        if TABULATE_AVAILABLE and data_dicts:
            # Select key fields for table display
            table_data = []
            for item in data_dicts:
                table_data.append({
                    'ID': item.get('video_id', '')[:11],
                    'Title': item.get('title', '')[:50],
                    'Channel': item.get('channel_name', '')[:30],
                    'Views': item.get('view_count', 'N/A'),
                    'Upload Date': str(item.get('upload_date', ''))[:10]
                })
            
            if console and RICH_AVAILABLE:
                table = Table(title="YouTube Search Results")
                table.add_column("ID", style="cyan")
                table.add_column("Title", style="green")
                table.add_column("Channel", style="yellow") 
                table.add_column("Views", style="magenta")
                table.add_column("Upload Date", style="blue")
                
                for item in table_data:
                    table.add_row(
                        item['ID'],
                        item['Title'],
                        item['Channel'],
                        str(item['Views']),
                        item['Upload Date']
                    )
                
                console.print(table)
            else:
                print(tabulate(table_data, headers='keys', tablefmt='grid'))
        else:
            print("Table format requires 'tabulate' library")
    
    elif format_type == 'csv' and PANDAS_AVAILABLE:
        df = pd.DataFrame(data_dicts)
        if output_path:
            df.to_csv(output_path, index=False)
        else:
            print(df.to_csv(index=False))
    
    elif format_type == 'excel' and PANDAS_AVAILABLE:
        df = pd.DataFrame(data_dicts)
        output_file = output_path or 'youtube_results.xlsx'
        df.to_excel(output_file, index=False)
        if not quiet:
            if console:
                console.print(f"[green]Results saved to: {output_file}[/green]")
            else:
                print(f"Results saved to: {output_file}")
    
    else:
        print(json.dumps(data_dicts, indent=2, default=str))


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        if console:
            console.print("\n[yellow]Received interrupt signal. Shutting down gracefully...[/yellow]")
        else:
            print("\nReceived interrupt signal. Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == '__main__':
    setup_signal_handlers()
    
    # Check if we're in an async context
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're here, we're already in an async context
        main()
    except RuntimeError:
        # No event loop running, we can start our own
        asyncio.run(main())