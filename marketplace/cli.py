#!/usr/bin/env python3
"""
APG Capability Marketplace CLI
==============================

Command-line interface for interacting with the capability marketplace.
"""

import asyncio
import json
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.markdown import Markdown

from capability_marketplace import (
	CapabilityMarketplace,
	MarketplaceCapability,
	CapabilityCategory,
	CapabilityStatus,
	LicenseType,
	CapabilityDependency,
	CapabilityRating
)

# Rich console for beautiful output
console = Console()

class MarketplaceCLI:
	"""CLI interface for the capability marketplace"""
	
	def __init__(self, marketplace_path: str = "./marketplace_data"):
		self.marketplace_path = marketplace_path
		self.marketplace: Optional[CapabilityMarketplace] = None
		self.logger = logging.getLogger("marketplace_cli")
	
	async def init_marketplace(self):
		"""Initialize marketplace connection"""
		if self.marketplace is None:
			self.marketplace = CapabilityMarketplace(self.marketplace_path)
			await asyncio.sleep(0.1)  # Allow initialization
	
	async def list_capabilities(
		self, 
		category: Optional[str] = None,
		status: Optional[str] = None,
		author: Optional[str] = None,
		format_output: str = "table"
	):
		"""List available capabilities"""
		await self.init_marketplace()
		
		# Convert filters
		category_filter = None
		if category:
			try:
				category_filter = CapabilityCategory(category)
			except ValueError:
				console.print(f"[red]Invalid category: {category}[/red]")
				return
		
		status_filter = None
		if status:
			try:
				status_filter = CapabilityStatus(status)
			except ValueError:
				console.print(f"[red]Invalid status: {status}[/red]")
				return
		
		capabilities = await self.marketplace.list_capabilities(
			status=status_filter,
			category=category_filter,
			author=author
		)
		
		if not capabilities:
			console.print("[yellow]No capabilities found matching the criteria[/yellow]")
			return
		
		if format_output == "json":
			data = []
			for cap in capabilities:
				data.append({
					"id": cap.id,
					"name": cap.name,
					"description": cap.description,
					"category": cap.category.value,
					"author": cap.author,
					"status": cap.status.value,
					"rating": cap.metrics.average_rating,
					"downloads": cap.metrics.download_count
				})
			console.print(json.dumps(data, indent=2))
		else:
			# Table format
			table = Table(title="APG Capabilities")
			table.add_column("Name", style="cyan", no_wrap=True)
			table.add_column("Category", style="green")
			table.add_column("Author", style="blue")
			table.add_column("Status", style="yellow")
			table.add_column("Rating", justify="right", style="magenta")
			table.add_column("Downloads", justify="right", style="red")
			
			for cap in capabilities:
				rating_str = f"{cap.metrics.average_rating:.1f} ⭐" if cap.metrics.rating_count > 0 else "No ratings"
				table.add_row(
					cap.name,
					cap.category.value.replace('_', ' ').title(),
					cap.author,
					cap.status.value.title(),
					rating_str,
					str(cap.metrics.download_count)
				)
			
			console.print(table)
	
	async def search_capabilities(self, query: str, category: Optional[str] = None, max_results: int = 20):
		"""Search for capabilities"""
		await self.init_marketplace()
		
		category_filter = None
		if category:
			try:
				category_filter = CapabilityCategory(category)
			except ValueError:
				console.print(f"[red]Invalid category: {category}[/red]")
				return
		
		with Progress(
			SpinnerColumn(),
			TextColumn("[progress.description]{task.description}"),
			console=console
		) as progress:
			task = progress.add_task("Searching capabilities...", total=None)
			
			capabilities = await self.marketplace.search_capabilities(
				query=query,
				category=category_filter,
				max_results=max_results
			)
			
			progress.remove_task(task)
		
		if not capabilities:
			console.print(f"[yellow]No capabilities found for query: '{query}'[/yellow]")
			return
		
		console.print(f"\n[bold]Found {len(capabilities)} capabilities:[/bold]\n")
		
		for i, cap in enumerate(capabilities, 1):
			panel_content = f"""
**Category:** {cap.category.value.replace('_', ' ').title()}
**Author:** {cap.author}
**Status:** {cap.status.value.title()}
**Rating:** {cap.metrics.average_rating:.1f} ⭐ ({cap.metrics.rating_count} reviews)
**Downloads:** {cap.metrics.download_count}

{cap.description}
			""".strip()
			
			console.print(Panel(
				panel_content,
				title=f"{i}. {cap.name}",
				title_align="left",
				border_style="blue"
			))
	
	async def show_capability(self, capability_id: str, include_code: bool = False):
		"""Show detailed information about a capability"""
		await self.init_marketplace()
		
		capability = await self.marketplace.get_capability(capability_id)
		if not capability:
			console.print(f"[red]Capability not found: {capability_id}[/red]")
			return
		
		# Basic information
		info_content = f"""
**ID:** {capability.id}
**Display Name:** {capability.display_name}
**Category:** {capability.category.value.replace('_', ' ').title()}
**Author:** {capability.author} ({capability.author_email})
**Organization:** {capability.organization or 'None'}
**License:** {capability.license.value.upper()}
**Version:** {capability.current_version}
**Status:** {capability.status.value.title()}

**Rating:** {capability.metrics.average_rating:.1f} ⭐ ({capability.metrics.rating_count} reviews)
**Downloads:** {capability.metrics.download_count}
**Created:** {capability.created_at.strftime('%Y-%m-%d %H:%M')}
**Updated:** {capability.updated_at.strftime('%Y-%m-%d %H:%M')}

**Tags:** {', '.join(capability.tags) if capability.tags else 'None'}
**Keywords:** {', '.join(capability.keywords) if capability.keywords else 'None'}
**Platforms:** {', '.join(capability.platforms)}

**Repository:** {capability.repository or 'None'}
**Homepage:** {capability.homepage or 'None'}
		""".strip()
		
		console.print(Panel(info_content, title=capability.name, border_style="green"))
		
		# Description
		if capability.detailed_description:
			console.print(Panel(
				Markdown(capability.detailed_description),
				title="Description",
				border_style="blue"
			))
		
		# Dependencies
		if capability.dependencies:
			dep_table = Table(title="Dependencies")
			dep_table.add_column("Name", style="cyan")
			dep_table.add_column("Version", style="green")
			dep_table.add_column("Optional", style="yellow")
			dep_table.add_column("Description", style="white")
			
			for dep in capability.dependencies:
				dep_table.add_row(
					dep.name,
					dep.version_constraint,
					"Yes" if dep.optional else "No",
					dep.description or ""
				)
			
			console.print(dep_table)
		
		# Example usage
		if capability.example_usage:
			console.print(Panel(
				f"```python\n{capability.example_usage}\n```",
				title="Example Usage",
				border_style="yellow"
			))
		
		# Code (if requested)
		if include_code and capability.capability_code:
			console.print(Panel(
				f"```python\n{capability.capability_code[:1000]}{'...' if len(capability.capability_code) > 1000 else ''}\n```",
				title="Capability Code (first 1000 chars)",
				border_style="red"
			))
	
	async def download_capability(self, capability_id: str, output_dir: str = "./downloaded_capabilities", user_id: str = "cli_user"):
		"""Download a capability"""
		await self.init_marketplace()
		
		with Progress(
			SpinnerColumn(),
			TextColumn("[progress.description]{task.description}"),
			console=console
		) as progress:
			task = progress.add_task("Downloading capability...", total=None)
			
			package = await self.marketplace.download_capability(capability_id, user_id)
			
			progress.remove_task(task)
		
		if not package:
			console.print(f"[red]Failed to download capability: {capability_id}[/red]")
			return
		
		# Create output directory
		output_path = Path(output_dir)
		output_path.mkdir(exist_ok=True)
		
		capability = package['capability']
		cap_dir = output_path / f"{capability.name}_{capability.current_version}"
		cap_dir.mkdir(exist_ok=True)
		
		# Save capability files
		files_created = []
		
		# Main capability code
		if package['code']:
			code_file = cap_dir / f"{capability.name}.py"
			code_file.write_text(package['code'])
			files_created.append(str(code_file))
		
		# Documentation
		if package['documentation']:
			docs_file = cap_dir / "README.md"
			docs_file.write_text(package['documentation'])
			files_created.append(str(docs_file))
		
		# Example usage
		if package['example_usage']:
			example_file = cap_dir / "example.py"
			example_file.write_text(package['example_usage'])
			files_created.append(str(example_file))
		
		# Test cases
		if package['test_cases']:
			tests_dir = cap_dir / "tests"
			tests_dir.mkdir(exist_ok=True)
			for i, test in enumerate(package['test_cases']):
				test_file = tests_dir / f"test_{i+1}.py"
				test_file.write_text(test)
				files_created.append(str(test_file))
		
		# Dependencies file
		if package['dependencies']:
			deps_data = []
			for dep in package['dependencies']:
				deps_data.append({
					"name": dep.name,
					"version_constraint": dep.version_constraint,
					"optional": dep.optional,
					"description": dep.description
				})
			
			deps_file = cap_dir / "dependencies.json"
			deps_file.write_text(json.dumps(deps_data, indent=2))
			files_created.append(str(deps_file))
		
		# Capability metadata
		metadata = {
			"id": capability.id,
			"name": capability.name,
			"version": capability.current_version,
			"author": capability.author,
			"license": capability.license.value,
			"category": capability.category.value,
			"download_date": asyncio.get_event_loop().time()
		}
		
		metadata_file = cap_dir / "capability.json"
		metadata_file.write_text(json.dumps(metadata, indent=2))
		files_created.append(str(metadata_file))
		
		console.print(f"[green]✓ Successfully downloaded capability to: {cap_dir}[/green]")
		console.print(f"[blue]Files created:[/blue]")
		for file_path in files_created:
			console.print(f"  • {file_path}")
	
	async def submit_capability(self, capability_file: str):
		"""Submit a capability from a JSON file"""
		await self.init_marketplace()
		
		try:
			with open(capability_file, 'r') as f:
				data = json.load(f)
		except FileNotFoundError:
			console.print(f"[red]File not found: {capability_file}[/red]")
			return
		except json.JSONDecodeError as e:
			console.print(f"[red]Invalid JSON in file: {e}[/red]")
			return
		
		try:
			# Convert dependencies
			dependencies = []
			for dep_data in data.get('dependencies', []):
				dependencies.append(CapabilityDependency(
					name=dep_data.get('name', ''),
					version_constraint=dep_data.get('version_constraint', '*'),
					optional=dep_data.get('optional', False),
					description=dep_data.get('description', '')
				))
			
			# Create capability
			capability = MarketplaceCapability(
				name=data['name'],
				display_name=data.get('display_name', data['name']),
				description=data['description'],
				detailed_description=data.get('detailed_description', ''),
				category=CapabilityCategory(data.get('category', 'custom')),
				tags=data.get('tags', []),
				keywords=data.get('keywords', []),
				author=data['author'],
				author_email=data['author_email'],
				organization=data.get('organization', ''),
				license=LicenseType(data.get('license', 'mit')),
				homepage=data.get('homepage', ''),
				repository=data.get('repository', ''),
				capability_code=data['capability_code'],
				example_usage=data.get('example_usage', ''),
				documentation=data.get('documentation', ''),
				dependencies=dependencies,
				platforms=data.get('platforms', ['linux', 'windows', 'macos'])
			)
			
			with Progress(
				SpinnerColumn(),
				TextColumn("[progress.description]{task.description}"),
				console=console
			) as progress:
				task = progress.add_task("Submitting capability...", total=None)
				
				result = await self.marketplace.submit_capability(capability)
				
				progress.remove_task(task)
			
			if result['success']:
				console.print(f"[green]✓ Successfully submitted capability: {capability.name}[/green]")
				console.print(f"[blue]Capability ID: {result['capability_id']}[/blue]")
				console.print(f"[blue]Validation Score: {result['validation_results']['score']:.1f}/100[/blue]")
				
				if result['validation_results']['warnings']:
					console.print("[yellow]Warnings:[/yellow]")
					for warning in result['validation_results']['warnings']:
						console.print(f"  • {warning}")
			else:
				console.print(f"[red]✗ Failed to submit capability[/red]")
				console.print(f"[red]Errors:[/red]")
				for error in result['errors']:
					console.print(f"  • {error}")
		
		except KeyError as e:
			console.print(f"[red]Missing required field in JSON: {e}[/red]")
		except ValueError as e:
			console.print(f"[red]Invalid value in JSON: {e}[/red]")
		except Exception as e:
			console.print(f"[red]Error submitting capability: {e}[/red]")
	
	async def publish_capability(self, capability_id: str):
		"""Publish a capability"""
		await self.init_marketplace()
		
		capability = await self.marketplace.get_capability(capability_id)
		if not capability:
			console.print(f"[red]Capability not found: {capability_id}[/red]")
			return
		
		console.print(f"Publishing capability: {capability.name}")
		
		if not Confirm.ask("Are you sure you want to publish this capability?"):
			console.print("[yellow]Publishing cancelled[/yellow]")
			return
		
		with Progress(
			SpinnerColumn(),
			TextColumn("[progress.description]{task.description}"),
			console=console
		) as progress:
			task = progress.add_task("Publishing capability...", total=None)
			
			success = await self.marketplace.publish_capability(capability_id)
			
			progress.remove_task(task)
		
		if success:
			console.print(f"[green]✓ Successfully published capability: {capability.name}[/green]")
		else:
			console.print(f"[red]✗ Failed to publish capability. Check validation requirements.[/red]")
	
	async def add_rating(self, capability_id: str, rating: int, review: str = "", user_id: str = "cli_user"):
		"""Add a rating to a capability"""
		await self.init_marketplace()
		
		if not (1 <= rating <= 5):
			console.print("[red]Rating must be between 1 and 5 stars[/red]")
			return
		
		capability = await self.marketplace.get_capability(capability_id)
		if not capability:
			console.print(f"[red]Capability not found: {capability_id}[/red]")
			return
		
		rating_obj = CapabilityRating(
			user_id=user_id,
			capability_id=capability_id,
			rating=rating,
			review=review
		)
		
		success = await self.marketplace.add_rating(rating_obj)
		
		if success:
			console.print(f"[green]✓ Added {rating} star rating for {capability.name}[/green]")
		else:
			console.print(f"[red]✗ Failed to add rating[/red]")
	
	async def show_stats(self):
		"""Show marketplace statistics"""
		await self.init_marketplace()
		
		stats = self.marketplace.get_marketplace_stats()
		
		# Overview stats
		overview_content = f"""
**Total Capabilities:** {stats['total_capabilities']}
**Published:** {stats['published_capabilities']}
**Verified:** {stats['verified_capabilities']}
**Total Downloads:** {stats['total_downloads']}
		""".strip()
		
		console.print(Panel(overview_content, title="Marketplace Overview", border_style="green"))
		
		# Category breakdown
		if stats['categories']:
			cat_table = Table(title="Categories")
			cat_table.add_column("Category", style="cyan")
			cat_table.add_column("Count", justify="right", style="green")
			
			for category, count in stats['categories'].items():
				cat_table.add_row(category.replace('_', ' ').title(), str(count))
			
			console.print(cat_table)
		
		# Top authors
		if stats['top_authors']:
			author_table = Table(title="Top Authors")
			author_table.add_column("Author", style="blue")
			author_table.add_column("Capabilities", justify="right", style="green")
			
			for author, count in list(stats['top_authors'].items())[:10]:
				author_table.add_row(author, str(count))
			
			console.print(author_table)
		
		# Recent activity
		if stats['recent_activity']:
			activity_table = Table(title="Recent Activity")
			activity_table.add_column("Capability", style="cyan")
			activity_table.add_column("Author", style="blue")
			activity_table.add_column("Action", style="green")
			activity_table.add_column("Time", style="yellow")
			
			for activity in stats['recent_activity'][:10]:
				activity_table.add_row(
					activity['capability_name'],
					activity['author'],
					activity['action'].title(),
					activity['timestamp'][:16].replace('T', ' ')
				)
			
			console.print(activity_table)

# CLI Command Interface

@click.group()
@click.option('--marketplace-path', default='./marketplace_data', help='Path to marketplace data directory')
@click.pass_context
def cli(ctx, marketplace_path):
	"""APG Capability Marketplace CLI"""
	ctx.ensure_object(dict)
	ctx.obj['cli'] = MarketplaceCLI(marketplace_path)

@cli.command()
@click.option('--category', help='Filter by category')
@click.option('--status', help='Filter by status (draft, published, verified, etc.)')
@click.option('--author', help='Filter by author')
@click.option('--format', 'format_output', default='table', type=click.Choice(['table', 'json']), help='Output format')
@click.pass_context
def list(ctx, category, status, author, format_output):
	"""List available capabilities"""
	cli_instance = ctx.obj['cli']
	asyncio.run(cli_instance.list_capabilities(category, status, author, format_output))

@cli.command()
@click.argument('query')
@click.option('--category', help='Filter by category')
@click.option('--max-results', default=20, help='Maximum number of results')
@click.pass_context
def search(ctx, query, category, max_results):
	"""Search for capabilities"""
	cli_instance = ctx.obj['cli']
	asyncio.run(cli_instance.search_capabilities(query, category, max_results))

@cli.command()
@click.argument('capability_id')
@click.option('--include-code', is_flag=True, help='Include capability code in output')
@click.pass_context
def show(ctx, capability_id, include_code):
	"""Show detailed information about a capability"""
	cli_instance = ctx.obj['cli']
	asyncio.run(cli_instance.show_capability(capability_id, include_code))

@cli.command()
@click.argument('capability_id')
@click.option('--output-dir', default='./downloaded_capabilities', help='Output directory')
@click.option('--user-id', default='cli_user', help='User ID for download tracking')
@click.pass_context
def download(ctx, capability_id, output_dir, user_id):
	"""Download a capability"""
	cli_instance = ctx.obj['cli']
	asyncio.run(cli_instance.download_capability(capability_id, output_dir, user_id))

@cli.command()
@click.argument('capability_file')
@click.pass_context
def submit(ctx, capability_file):
	"""Submit a capability from a JSON file"""
	cli_instance = ctx.obj['cli']
	asyncio.run(cli_instance.submit_capability(capability_file))

@cli.command()
@click.argument('capability_id')
@click.pass_context
def publish(ctx, capability_id):
	"""Publish a capability"""
	cli_instance = ctx.obj['cli']
	asyncio.run(cli_instance.publish_capability(capability_id))

@cli.command()
@click.argument('capability_id')
@click.argument('rating', type=int)
@click.option('--review', default='', help='Review text')
@click.option('--user-id', default='cli_user', help='User ID')
@click.pass_context
def rate(ctx, capability_id, rating, review, user_id):
	"""Add a rating to a capability (1-5 stars)"""
	cli_instance = ctx.obj['cli']
	asyncio.run(cli_instance.add_rating(capability_id, rating, review, user_id))

@cli.command()
@click.pass_context
def stats(ctx):
	"""Show marketplace statistics"""
	cli_instance = ctx.obj['cli']
	asyncio.run(cli_instance.show_stats())

@cli.command()
def categories():
	"""List available categories"""
	table = Table(title="Available Categories")
	table.add_column("Value", style="cyan")
	table.add_column("Name", style="green")
	
	for category in CapabilityCategory:
		table.add_row(category.value, category.name.replace('_', ' ').title())
	
	console.print(table)

@cli.command()
def licenses():
	"""List available license types"""
	table = Table(title="Available Licenses")
	table.add_column("Value", style="cyan")
	table.add_column("Name", style="green")
	
	for license_type in LicenseType:
		table.add_row(license_type.value, license_type.name.replace('_', ' ').title())
	
	console.print(table)

if __name__ == '__main__':
	cli()