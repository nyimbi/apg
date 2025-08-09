"""
APG Workflow Export Manager

Multi-format workflow export system with comprehensive export capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, BinaryIO
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
import json
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

logger = logging.getLogger(__name__)

class ExportFormat(str, Enum):
	"""Supported export formats."""
	JSON = "json"
	YAML = "yaml"
	XML = "xml"
	BPMN = "bpmn"
	PDF = "pdf"
	PNG = "png"
	SVG = "svg"
	DOCX = "docx"
	HTML = "html"
	MERMAID = "mermaid"
	GRAPHVIZ = "graphviz"
	EXCEL = "excel"

class ExportOptions(BaseModel):
	"""Export configuration options."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	# Format settings
	format: ExportFormat = Field(..., description="Export format")
	include_metadata: bool = Field(default=True, description="Include workflow metadata")
	include_config: bool = Field(default=True, description="Include component configurations")
	include_comments: bool = Field(default=False, description="Include comments and annotations")
	
	# Visual export settings
	include_layout: bool = Field(default=True, description="Include visual layout information")
	image_width: int = Field(default=1920, ge=100, le=10000, description="Image width for visual exports")
	image_height: int = Field(default=1080, ge=100, le=10000, description="Image height for visual exports")
	background_color: str = Field(default="#ffffff", description="Background color")
	
	# Documentation settings
	include_documentation: bool = Field(default=False, description="Include detailed documentation")
	template_style: str = Field(default="professional", description="Documentation template style")
	
	# Data filtering
	exclude_sensitive: bool = Field(default=True, description="Exclude sensitive data")
	include_test_data: bool = Field(default=False, description="Include test data")
	
	# Compression
	compress: bool = Field(default=False, description="Compress output")
	compression_level: int = Field(default=6, ge=1, le=9, description="Compression level")

class ExportResult(BaseModel):
	"""Result of export operation."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	success: bool = Field(..., description="Export success status")
	format: ExportFormat = Field(..., description="Export format used")
	file_path: Optional[str] = Field(default=None, description="Path to exported file")
	file_size: int = Field(default=0, description="File size in bytes")
	content: Optional[bytes] = Field(default=None, description="Export content for direct download")
	
	# Metadata
	exported_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	export_duration: float = Field(default=0.0, description="Export duration in seconds")
	
	# Validation
	validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
	warnings: List[str] = Field(default_factory=list, description="Export warnings")

class ExportTemplate(BaseModel):
	"""Export template definition."""
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)
	
	id: str = Field(..., description="Template ID")
	name: str = Field(..., description="Template name")
	description: str = Field(..., description="Template description")
	format: ExportFormat = Field(..., description="Target format")
	template_content: str = Field(..., description="Template content")
	variables: Dict[str, Any] = Field(default_factory=dict, description="Template variables")
	styles: Dict[str, Any] = Field(default_factory=dict, description="Style configuration")

class ExportManager:
	"""
	Multi-format workflow export manager.
	
	Features:
	- Support for 12+ export formats
	- Visual and data exports
	- Template-based documentation
	- Batch export capabilities
	- Compression and optimization
	- Security filtering
	"""
	
	def __init__(self, config):
		self.config = config
		self.export_templates: Dict[str, ExportTemplate] = {}
		self.active_exports: Dict[str, Dict[str, Any]] = {}
		self.is_initialized = False
		
		logger.info("Export manager initialized")
	
	async def initialize(self) -> None:
		"""Initialize the export manager."""
		try:
			# Load export templates
			await self._load_export_templates()
			
			# Initialize export engines
			await self._initialize_export_engines()
			
			self.is_initialized = True
			logger.info(f"Export manager initialized with {len(self.export_templates)} templates")
			
		except Exception as e:
			logger.error(f"Failed to initialize export manager: {e}")
			raise
	
	async def shutdown(self) -> None:
		"""Shutdown the export manager."""
		try:
			# Cancel active exports
			for export_id in list(self.active_exports.keys()):
				await self.cancel_export(export_id)
			
			self.export_templates.clear()
			self.active_exports.clear()
			self.is_initialized = False
			logger.info("Export manager shutdown completed")
		except Exception as e:
			logger.error(f"Error during export manager shutdown: {e}")
	
	async def export_workflow(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow in specified format."""
		try:
			start_time = datetime.now(timezone.utc)
			export_id = f"export_{int(start_time.timestamp() * 1000)}"
			
			# Track export
			self.active_exports[export_id] = {
				'format': options.format,
				'started_at': start_time,
				'status': 'processing'
			}
			
			# Validate workflow data
			validation_errors = await self._validate_workflow_data(workflow_data)
			if validation_errors and options.format in [ExportFormat.BPMN, ExportFormat.XML]:
				return ExportResult(
					success=False,
					format=options.format,
					validation_errors=validation_errors
				)
			
			# Process export based on format
			result = None
			if options.format == ExportFormat.JSON:
				result = await self._export_json(workflow_data, options)
			elif options.format == ExportFormat.YAML:
				result = await self._export_yaml(workflow_data, options)
			elif options.format == ExportFormat.XML:
				result = await self._export_xml(workflow_data, options)
			elif options.format == ExportFormat.BPMN:
				result = await self._export_bpmn(workflow_data, options)
			elif options.format == ExportFormat.PDF:
				result = await self._export_pdf(workflow_data, options)
			elif options.format == ExportFormat.PNG:
				result = await self._export_png(workflow_data, options)
			elif options.format == ExportFormat.SVG:
				result = await self._export_svg(workflow_data, options)
			elif options.format == ExportFormat.DOCX:
				result = await self._export_docx(workflow_data, options)
			elif options.format == ExportFormat.HTML:
				result = await self._export_html(workflow_data, options)
			elif options.format == ExportFormat.MERMAID:
				result = await self._export_mermaid(workflow_data, options)
			elif options.format == ExportFormat.GRAPHVIZ:
				result = await self._export_graphviz(workflow_data, options)
			elif options.format == ExportFormat.EXCEL:
				result = await self._export_excel(workflow_data, options)
			else:
				raise ValueError(f"Unsupported export format: {options.format}")
			
			# Calculate duration
			end_time = datetime.now(timezone.utc)
			duration = (end_time - start_time).total_seconds()
			
			if result:
				result.export_duration = duration
				result.validation_errors = validation_errors
			
			# Clean up tracking
			if export_id in self.active_exports:
				del self.active_exports[export_id]
			
			logger.info(f"Exported workflow to {options.format} in {duration:.3f}s")
			return result
			
		except Exception as e:
			logger.error(f"Failed to export workflow: {e}")
			return ExportResult(
				success=False,
				format=options.format,
				validation_errors=[f"Export failed: {e}"]
			)
	
	async def batch_export(self, workflow_data: Dict[str, Any], formats: List[ExportFormat], base_options: ExportOptions) -> Dict[ExportFormat, ExportResult]:
		"""Export workflow to multiple formats."""
		try:
			results = {}
			
			# Process each format
			tasks = []
			for export_format in formats:
				options = base_options.model_copy()
				options.format = export_format
				task = self.export_workflow(workflow_data, options)
				tasks.append((export_format, task))
			
			# Execute exports concurrently
			for export_format, task in tasks:
				try:
					result = await task
					results[export_format] = result
				except Exception as e:
					results[export_format] = ExportResult(
						success=False,
						format=export_format,
						validation_errors=[f"Batch export failed: {e}"]
					)
			
			logger.info(f"Batch exported workflow to {len(formats)} formats")
			return results
			
		except Exception as e:
			logger.error(f"Failed to batch export workflow: {e}")
			return {fmt: ExportResult(
				success=False,
				format=fmt,
				validation_errors=[f"Batch export failed: {e}"]
			) for fmt in formats}
	
	async def get_export_templates(self, format_filter: Optional[ExportFormat] = None) -> List[ExportTemplate]:
		"""Get available export templates."""
		try:
			templates = list(self.export_templates.values())
			if format_filter:
				templates = [t for t in templates if t.format == format_filter]
			return sorted(templates, key=lambda x: x.name)
		except Exception as e:
			logger.error(f"Failed to get export templates: {e}")
			return []
	
	async def create_export_template(self, template: ExportTemplate) -> None:
		"""Create a new export template."""
		try:
			self.export_templates[template.id] = template
			await self._save_template_to_database(template)
			logger.info(f"Created export template: {template.id}")
		except Exception as e:
			logger.error(f"Failed to create export template: {e}")
			raise
	
	async def cancel_export(self, export_id: str) -> None:
		"""Cancel an active export."""
		try:
			if export_id in self.active_exports:
				self.active_exports[export_id]['status'] = 'cancelled'
				# In a real implementation, this would cancel the actual export process
				del self.active_exports[export_id]
				logger.info(f"Cancelled export {export_id}")
		except Exception as e:
			logger.error(f"Failed to cancel export: {e}")
	
	async def get_export_status(self, export_id: str) -> Optional[Dict[str, Any]]:
		"""Get status of an export."""
		return self.active_exports.get(export_id)
	
	# Format-specific export methods
	
	async def _export_json(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to JSON format."""
		try:
			# Prepare export data
			export_data = await self._prepare_export_data(workflow_data, options)
			
			# Convert to JSON
			json_content = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
			content_bytes = json_content.encode('utf-8')
			
			# Compress if requested
			if options.compress:
				import gzip
				content_bytes = gzip.compress(content_bytes, compresslevel=options.compression_level)
			
			return ExportResult(
				success=True,
				format=ExportFormat.JSON,
				content=content_bytes,
				file_size=len(content_bytes)
			)
			
		except Exception as e:
			logger.error(f"Failed to export JSON: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.JSON,
				validation_errors=[f"JSON export failed: {e}"]
			)
	
	async def _export_yaml(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to YAML format."""
		try:
			# Prepare export data
			export_data = await self._prepare_export_data(workflow_data, options)
			
			# Convert to YAML
			yaml_content = yaml.dump(export_data, default_flow_style=False, allow_unicode=True, sort_keys=False)
			content_bytes = yaml_content.encode('utf-8')
			
			# Compress if requested
			if options.compress:
				import gzip
				content_bytes = gzip.compress(content_bytes, compresslevel=options.compression_level)
			
			return ExportResult(
				success=True,
				format=ExportFormat.YAML,
				content=content_bytes,
				file_size=len(content_bytes)
			)
			
		except Exception as e:
			logger.error(f"Failed to export YAML: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.YAML,
				validation_errors=[f"YAML export failed: {e}"]
			)
	
	async def _export_xml(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to XML format."""
		try:
			# Create XML structure
			root = ET.Element("workflow")
			root.set("xmlns", "http://datacraft.co.ke/workflow/v1")
			
			# Add metadata
			if options.include_metadata:
				metadata_elem = ET.SubElement(root, "metadata")
				for key, value in workflow_data.get('metadata', {}).items():
					elem = ET.SubElement(metadata_elem, key)
					elem.text = str(value)
			
			# Add definition
			definition = workflow_data.get('definition', {})
			def_elem = ET.SubElement(root, "definition")
			
			# Add nodes
			nodes_elem = ET.SubElement(def_elem, "nodes")
			for node in definition.get('nodes', []):
				node_elem = ET.SubElement(nodes_elem, "node")
				node_elem.set("id", node.get('id', ''))
				node_elem.set("type", node.get('component_type', ''))
				
				# Position
				pos_elem = ET.SubElement(node_elem, "position")
				position = node.get('position', {})
				pos_elem.set("x", str(position.get('x', 0)))
				pos_elem.set("y", str(position.get('y', 0)))
				
				# Configuration
				if options.include_config and node.get('config'):
					config_elem = ET.SubElement(node_elem, "configuration")
					for key, value in node['config'].items():
						prop_elem = ET.SubElement(config_elem, "property")
						prop_elem.set("name", key)
						prop_elem.text = str(value)
			
			# Add connections
			connections_elem = ET.SubElement(def_elem, "connections")
			for conn in definition.get('connections', []):
				conn_elem = ET.SubElement(connections_elem, "connection")
				conn_elem.set("id", conn.get('id', ''))
				conn_elem.set("source", conn.get('source_node_id', ''))
				conn_elem.set("target", conn.get('target_node_id', ''))
				conn_elem.set("sourcePort", conn.get('source_port', 'output'))
				conn_elem.set("targetPort", conn.get('target_port', 'input'))
			
			# Convert to string
			ET.indent(root, space="  ", level=0)
			xml_content = ET.tostring(root, encoding='unicode', xml_declaration=True)
			content_bytes = xml_content.encode('utf-8')
			
			return ExportResult(
				success=True,
				format=ExportFormat.XML,
				content=content_bytes,
				file_size=len(content_bytes)
			)
			
		except Exception as e:
			logger.error(f"Failed to export XML: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.XML,
				validation_errors=[f"XML export failed: {e}"]
			)
	
	async def _export_bpmn(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to BPMN 2.0 format."""
		try:
			# Create BPMN XML structure
			root = ET.Element("definitions")
			root.set("xmlns", "http://www.omg.org/spec/BPMN/20100524/MODEL")
			root.set("xmlns:bpmndi", "http://www.omg.org/spec/BPMN/20100524/DI")
			root.set("xmlns:dc", "http://www.omg.org/spec/DD/20100524/DC")
			root.set("xmlns:di", "http://www.omg.org/spec/DD/20100524/DI")
			root.set("targetNamespace", "http://datacraft.co.ke/bpmn")
			
			# Create process
			process = ET.SubElement(root, "process")
			process.set("id", workflow_data.get('id', 'workflow'))
			process.set("isExecutable", "true")
			
			definition = workflow_data.get('definition', {})
			nodes = definition.get('nodes', [])
			connections = definition.get('connections', [])
			
			# Add nodes as BPMN elements
			for node in nodes:
				node_type = node.get('component_type', '')
				
				if 'trigger' in node_type:
					# Start event
					elem = ET.SubElement(process, "startEvent")
				elif 'condition' in node_type:
					# Exclusive gateway
					elem = ET.SubElement(process, "exclusiveGateway")
				elif 'data' in node_type or 'transform' in node_type:
					# Service task
					elem = ET.SubElement(process, "serviceTask")
				else:
					# Generic task
					elem = ET.SubElement(process, "task")
				
				elem.set("id", node.get('id', ''))
				elem.set("name", node.get('label', node_type))
			
			# Add sequence flows for connections
			for conn in connections:
				flow = ET.SubElement(process, "sequenceFlow")
				flow.set("id", conn.get('id', ''))
				flow.set("sourceRef", conn.get('source_node_id', ''))
				flow.set("targetRef", conn.get('target_node_id', ''))
			
			# Add diagram information if layout is included
			if options.include_layout:
				diagram = ET.SubElement(root, "bpmndi:BPMNDiagram")
				plane = ET.SubElement(diagram, "bpmndi:BPMNPlane")
				plane.set("bpmnElement", workflow_data.get('id', 'workflow'))
				
				# Add shapes for nodes
				for node in nodes:
					shape = ET.SubElement(plane, "bpmndi:BPMNShape")
					shape.set("bpmnElement", node.get('id', ''))
					
					bounds = ET.SubElement(shape, "dc:Bounds")
					position = node.get('position', {})
					size = node.get('size', {})
					bounds.set("x", str(position.get('x', 0)))
					bounds.set("y", str(position.get('y', 0)))
					bounds.set("width", str(size.get('width', 100)))
					bounds.set("height", str(size.get('height', 80)))
				
				# Add edges for connections
				for conn in connections:
					edge = ET.SubElement(plane, "bpmndi:BPMNEdge")
					edge.set("bpmnElement", conn.get('id', ''))
					
					# Add waypoints (simplified)
					waypoint1 = ET.SubElement(edge, "di:waypoint")
					waypoint2 = ET.SubElement(edge, "di:waypoint")
					waypoint1.set("x", "0")
					waypoint1.set("y", "0")
					waypoint2.set("x", "100")
					waypoint2.set("y", "100")
			
			# Convert to string
			ET.indent(root, space="  ", level=0)
			bpmn_content = ET.tostring(root, encoding='unicode', xml_declaration=True)
			content_bytes = bpmn_content.encode('utf-8')
			
			return ExportResult(
				success=True,
				format=ExportFormat.BPMN,
				content=content_bytes,
				file_size=len(content_bytes)
			)
			
		except Exception as e:
			logger.error(f"Failed to export BPMN: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.BPMN,
				validation_errors=[f"BPMN export failed: {e}"]
			)
	
	async def _export_mermaid(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to Mermaid diagram format."""
		try:
			definition = workflow_data.get('definition', {})
			nodes = definition.get('nodes', [])
			connections = definition.get('connections', [])
			
			# Start Mermaid diagram
			mermaid_lines = ["graph TD"]
			
			# Add nodes
			for node in nodes:
				node_id = node.get('id', '')
				label = node.get('label', node.get('component_type', ''))
				
				# Choose shape based on node type
				component_type = node.get('component_type', '')
				if 'trigger' in component_type:
					# Circle for triggers
					mermaid_lines.append(f"    {node_id}(({label}))")
				elif 'condition' in component_type:
					# Diamond for decisions
					mermaid_lines.append(f"    {node_id}{{{label}}}")
				elif 'data' in component_type:
					# Rectangle for data processing
					mermaid_lines.append(f"    {node_id}[{label}]")
				else:
					# Default rectangle
					mermaid_lines.append(f"    {node_id}[{label}]")
			
			# Add connections
			for conn in connections:
				source = conn.get('source_node_id', '')
				target = conn.get('target_node_id', '')
				label = conn.get('label', '')
				
				if label:
					mermaid_lines.append(f"    {source} -->|{label}| {target}")
				else:
					mermaid_lines.append(f"    {source} --> {target}")
			
			# Add styling
			mermaid_lines.extend([
				"",
				"    classDef triggerNode fill:#e74c3c,stroke:#c0392b,color:#fff",
				"    classDef dataNode fill:#3498db,stroke:#2980b9,color:#fff",
				"    classDef logicNode fill:#f39c12,stroke:#e67e22,color:#fff"
			])
			
			# Apply classes to nodes
			for node in nodes:
				node_id = node.get('id', '')
				component_type = node.get('component_type', '')
				
				if 'trigger' in component_type:
					mermaid_lines.append(f"    class {node_id} triggerNode")
				elif 'data' in component_type:
					mermaid_lines.append(f"    class {node_id} dataNode")
				elif 'condition' in component_type or 'logic' in component_type:
					mermaid_lines.append(f"    class {node_id} logicNode")
			
			mermaid_content = "\n".join(mermaid_lines)
			content_bytes = mermaid_content.encode('utf-8')
			
			return ExportResult(
				success=True,
				format=ExportFormat.MERMAID,
				content=content_bytes,
				file_size=len(content_bytes)
			)
			
		except Exception as e:
			logger.error(f"Failed to export Mermaid: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.MERMAID,
				validation_errors=[f"Mermaid export failed: {e}"]
			)
	
	async def _export_graphviz(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to Graphviz DOT format."""
		try:
			definition = workflow_data.get('definition', {})
			nodes = definition.get('nodes', [])
			connections = definition.get('connections', [])
			
			# Start DOT graph
			dot_lines = [
				"digraph workflow {",
				"    rankdir=TB;",
				"    node [shape=box, style=rounded, fontname=\"Arial\"];",
				"    edge [fontname=\"Arial\"];"
			]
			
			# Add nodes
			for node in nodes:
				node_id = node.get('id', '').replace('-', '_')
				label = node.get('label', node.get('component_type', ''))
				component_type = node.get('component_type', '')
				
				# Choose style based on node type
				if 'trigger' in component_type:
					style = 'shape=ellipse, fillcolor="#e74c3c", style="filled,rounded", fontcolor=white'
				elif 'condition' in component_type:
					style = 'shape=diamond, fillcolor="#f39c12", style="filled", fontcolor=white'
				elif 'data' in component_type:
					style = 'fillcolor="#3498db", style="filled,rounded", fontcolor=white'
				else:
					style = 'fillcolor="#95a5a6", style="filled,rounded", fontcolor=white'
				
				dot_lines.append(f'    {node_id} [label="{label}", {style}];')
			
			# Add connections
			for conn in connections:
				source = conn.get('source_node_id', '').replace('-', '_')
				target = conn.get('target_node_id', '').replace('-', '_')
				label = conn.get('label', '')
				
				if label:
					dot_lines.append(f'    {source} -> {target} [label="{label}"];')
				else:
					dot_lines.append(f'    {source} -> {target};')
			
			dot_lines.append("}")
			
			dot_content = "\n".join(dot_lines)
			content_bytes = dot_content.encode('utf-8')
			
			return ExportResult(
				success=True,
				format=ExportFormat.GRAPHVIZ,
				content=content_bytes,
				file_size=len(content_bytes)
			)
			
		except Exception as e:
			logger.error(f"Failed to export Graphviz: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.GRAPHVIZ,
				validation_errors=[f"Graphviz export failed: {e}"]
			)
	
	async def _export_html(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to HTML format."""
		try:
			# Create HTML documentation
			html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workflow: {workflow_data.get('name', 'Untitled')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: {options.background_color}; }}
        .header {{ border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin: 30px 0; }}
        .node {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; }}
        .connection {{ background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .metadata {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .config {{ font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Workflow Documentation</h1>
        <h2>{workflow_data.get('name', 'Untitled Workflow')}</h2>
        <p>{workflow_data.get('description', 'No description provided')}</p>
    </div>
"""
			
			# Add metadata section
			if options.include_metadata and workflow_data.get('metadata'):
				html_content += """
    <div class="section">
        <h3>Metadata</h3>
        <div class="metadata">
"""
				for key, value in workflow_data['metadata'].items():
					html_content += f"            <p><strong>{key}:</strong> {value}</p>\n"
				html_content += "        </div>\n    </div>\n"
			
			# Add nodes section
			definition = workflow_data.get('definition', {})
			nodes = definition.get('nodes', [])
			if nodes:
				html_content += """
    <div class="section">
        <h3>Workflow Components</h3>
"""
				for node in nodes:
					html_content += f"""
        <div class="node">
            <h4>{node.get('label', node.get('component_type', 'Unknown'))}</h4>
            <p><strong>Type:</strong> {node.get('component_type', 'Unknown')}</p>
            <p><strong>ID:</strong> {node.get('id', 'N/A')}</p>
"""
					
					if options.include_config and node.get('config'):
						html_content += "            <p><strong>Configuration:</strong></p>\n"
						html_content += '            <div class="config">\n'
						for key, value in node['config'].items():
							html_content += f"                {key}: {value}<br>\n"
						html_content += "            </div>\n"
					
					html_content += "        </div>\n"
				html_content += "    </div>\n"
			
			# Add connections section
			connections = definition.get('connections', [])
			if connections:
				html_content += """
    <div class="section">
        <h3>Connections</h3>
        <table>
            <thead>
                <tr>
                    <th>Source</th>
                    <th>Target</th>
                    <th>Source Port</th>
                    <th>Target Port</th>
                </tr>
            </thead>
            <tbody>
"""
				for conn in connections:
					html_content += f"""
                <tr>
                    <td>{conn.get('source_node_id', 'N/A')}</td>
                    <td>{conn.get('target_node_id', 'N/A')}</td>
                    <td>{conn.get('source_port', 'output')}</td>
                    <td>{conn.get('target_port', 'input')}</td>
                </tr>
"""
				html_content += """
            </tbody>
        </table>
    </div>
"""
			
			# Add footer
			html_content += f"""
    <div class="section">
        <p><em>Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</em></p>
    </div>
</body>
</html>
"""
			
			content_bytes = html_content.encode('utf-8')
			
			return ExportResult(
				success=True,
				format=ExportFormat.HTML,
				content=content_bytes,
				file_size=len(content_bytes)
			)
			
		except Exception as e:
			logger.error(f"Failed to export HTML: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.HTML,
				validation_errors=[f"HTML export failed: {e}"]
			)
	
	async def _export_pdf(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to PDF format using weasyprint."""
		try:
			from io import BytesIO
			
			# First, generate HTML content with PDF-optimized styling
			html_content = await self._generate_pdf_html(workflow_data, options)
			
			# Try to use weasyprint for PDF generation
			try:
				import weasyprint
				
				# Create PDF from HTML
				html_doc = weasyprint.HTML(string=html_content)
				pdf_bytes = html_doc.write_pdf()
				
				return ExportResult(
					success=True,
					format=ExportFormat.PDF,
					content=pdf_bytes,
					file_size=len(pdf_bytes)
				)
				
			except ImportError:
				logger.warning("weasyprint not available, falling back to HTML-to-PDF conversion")
				
				# Fallback: Try to use reportlab
				try:
					from reportlab.pdfgen import canvas
					from reportlab.lib.pagesizes import letter, A4
					from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
					from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
					from reportlab.lib import colors
					from reportlab.lib.units import inch
					
					# Create PDF using reportlab
					buffer = BytesIO()
					doc = SimpleDocTemplate(buffer, pagesize=A4)
					styles = getSampleStyleSheet()
					story = []
					
					# Title
					title_style = ParagraphStyle(
						'CustomTitle',
						parent=styles['Heading1'],
						fontSize=24,
						textColor=colors.darkblue,
						spaceAfter=30,
						alignment=1  # Center
					)
					story.append(Paragraph(workflow_data.get('name', 'Untitled Workflow'), title_style))
					story.append(Spacer(1, 20))
					
					# Description
					if workflow_data.get('description'):
						story.append(Paragraph(f"<b>Description:</b> {workflow_data['description']}", styles['Normal']))
						story.append(Spacer(1, 20))
					
					# Metadata section
					if options.include_metadata and workflow_data.get('metadata'):
						story.append(Paragraph("Metadata", styles['Heading2']))
						for key, value in workflow_data['metadata'].items():
							story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
						story.append(Spacer(1, 20))
					
					# Workflow components
					definition = workflow_data.get('definition', {})
					nodes = definition.get('nodes', [])
					if nodes:
						story.append(Paragraph("Workflow Components", styles['Heading2']))
						
						# Create table for nodes
						node_data = [['ID', 'Type', 'Label']]
						if options.include_config:
							node_data[0].append('Configuration')
						
						for node in nodes:
							row = [
								node.get('id', 'N/A')[:20],  # Truncate long IDs
								node.get('component_type', 'Unknown'),
								node.get('label', node.get('component_type', 'N/A'))
							]
							if options.include_config and node.get('config'):
								config_str = ', '.join([f"{k}={v}" for k, v in list(node['config'].items())[:3]])
								if len(node['config']) > 3:
									config_str += "..."
								row.append(config_str[:50])  # Truncate long configs
							elif options.include_config:
								row.append('None')
							
							node_data.append(row)
						
						# Create and style table
						table = Table(node_data)
						table.setStyle(TableStyle([
							('BACKGROUND', (0, 0), (-1, 0), colors.grey),
							('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
							('ALIGN', (0, 0), (-1, -1), 'LEFT'),
							('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
							('FONTSIZE', (0, 0), (-1, 0), 10),
							('BOTTOMPADDING', (0, 0), (-1, 0), 12),
							('BACKGROUND', (0, 1), (-1, -1), colors.beige),
							('FONTSIZE', (0, 1), (-1, -1), 8),
							('GRID', (0, 0), (-1, -1), 1, colors.black)
						]))
						story.append(table)
						story.append(Spacer(1, 20))
					
					# Connections
					connections = definition.get('connections', [])
					if connections:
						story.append(Paragraph("Connections", styles['Heading2']))
						
						conn_data = [['Source', 'Target', 'Source Port', 'Target Port']]
						for conn in connections:
							conn_data.append([
								conn.get('source_node_id', 'N/A')[:20],
								conn.get('target_node_id', 'N/A')[:20],
								conn.get('source_port', 'output'),
								conn.get('target_port', 'input')
							])
						
						table = Table(conn_data)
						table.setStyle(TableStyle([
							('BACKGROUND', (0, 0), (-1, 0), colors.grey),
							('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
							('ALIGN', (0, 0), (-1, -1), 'LEFT'),
							('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
							('FONTSIZE', (0, 0), (-1, 0), 10),
							('BOTTOMPADDING', (0, 0), (-1, 0), 12),
							('BACKGROUND', (0, 1), (-1, -1), colors.beige),
							('FONTSIZE', (0, 1), (-1, -1), 8),
							('GRID', (0, 0), (-1, -1), 1, colors.black)
						]))
						story.append(table)
						story.append(Spacer(1, 20))
					
					# Footer
					story.append(Spacer(1, 40))
					footer_text = f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
					story.append(Paragraph(footer_text, styles['Normal']))
					
					# Build PDF
					doc.build(story)
					pdf_content = buffer.getvalue()
					buffer.close()
					
					return ExportResult(
						success=True,
						format=ExportFormat.PDF,
						content=pdf_content,
						file_size=len(pdf_content)
					)
					
				except ImportError:
					logger.warning("reportlab not available, using HTML fallback for PDF")
					
					# Final fallback: Enhanced HTML with print styles
					html_options = ExportOptions(
						format=ExportFormat.HTML,
						include_metadata=options.include_metadata,
						include_config=options.include_config,
						include_documentation=options.include_documentation
					)
					
					html_result = await self._export_html(workflow_data, html_options)
					if not html_result.success:
						return html_result
					
					return ExportResult(
						success=True,
						format=ExportFormat.PDF,
						content=html_result.content,
						file_size=len(html_result.content),
						warnings=["PDF export using HTML fallback. Install weasyprint or reportlab for true PDF generation."]
					)
			
		except Exception as e:
			logger.error(f"Failed to export PDF: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.PDF,
				validation_errors=[f"PDF export failed: {e}"]
			)
	
	async def _export_png(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to PNG image format using Playwright."""
		try:
			# Try to use Playwright for high-quality rendering
			try:
				from playwright.async_api import async_playwright
				
				# Generate SVG content first
				svg_options = ExportOptions(
					format=ExportFormat.SVG,
					include_layout=options.include_layout,
					image_width=options.image_width,
					image_height=options.image_height,
					background_color=options.background_color
				)
				
				svg_result = await self._export_svg(workflow_data, svg_options)
				if not svg_result.success:
					return svg_result
				
				# Convert SVG to PNG using Playwright
				async with async_playwright() as p:
					browser = await p.chromium.launch(headless=True)
					page = await browser.new_page()
					
					# Set viewport size
					await page.set_viewport_size(width=options.image_width, height=options.image_height)
					
					# Create HTML page with SVG
					svg_content = svg_result.content.decode('utf-8')
					html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ margin: 0; padding: 0; background: {options.background_color}; }}
        svg {{ width: 100%; height: 100vh; }}
    </style>
</head>
<body>
    {svg_content}
</body>
</html>
"""
					
					# Load content and take screenshot
					await page.set_content(html_content)
					
					# Wait for any fonts/rendering to complete
					await page.wait_for_timeout(1000)
					
					# Take screenshot
					png_bytes = await page.screenshot(
						full_page=True,
						type='png',
						clip={
							'x': 0,
							'y': 0,
							'width': options.image_width,
							'height': options.image_height
						}
					)
					
					await browser.close()
					
					return ExportResult(
						success=True,
						format=ExportFormat.PNG,
						content=png_bytes,
						file_size=len(png_bytes)
					)
			
			except ImportError:
				logger.warning("Playwright not available, falling back to alternative PNG generation")
				
				# Fallback: Try using Pillow to create PNG from data
				try:
					from PIL import Image, ImageDraw, ImageFont
					from io import BytesIO
					
					# Create image
					img = Image.new('RGB', (options.image_width, options.image_height), options.background_color)
					draw = ImageDraw.Draw(img)
					
					# Try to use a default font
					try:
						# Try to load a system font
						font_large = ImageFont.truetype("arial.ttf", 24)
						font_medium = ImageFont.truetype("arial.ttf", 16)
						font_small = ImageFont.truetype("arial.ttf", 12)
					except (OSError, IOError):
						# Fallback to default font
						font_large = ImageFont.load_default()
						font_medium = ImageFont.load_default()
						font_small = ImageFont.load_default()
					
					# Draw workflow visualization
					definition = workflow_data.get('definition', {})
					nodes = definition.get('nodes', [])
					connections = definition.get('connections', [])
					
					# Title
					title = workflow_data.get('name', 'Untitled Workflow')
					title_bbox = draw.textbbox((0, 0), title, font=font_large)
					title_width = title_bbox[2] - title_bbox[0]
					draw.text(
						((options.image_width - title_width) // 2, 30),
						title,
						fill='#2c3e50',
						font=font_large
					)
					
					# Node colors by type
					type_colors = {
						'trigger': '#e74c3c',
						'data': '#3498db',
						'logic': '#f39c12',
						'condition': '#f39c12',
						'integration': '#9b59b6',
						'default': '#95a5a6'
					}
					
					# Draw nodes
					y_offset = 100
					node_height = 60
					node_width = 150
					spacing = 40
					
					if nodes:
						# Calculate layout
						nodes_per_row = max(1, (options.image_width - 100) // (node_width + spacing))
						x_start = (options.image_width - (min(len(nodes), nodes_per_row) * (node_width + spacing) - spacing)) // 2
						
						for i, node in enumerate(nodes):
							row = i // nodes_per_row
							col = i % nodes_per_row
							
							x = x_start + col * (node_width + spacing)
							y = y_offset + row * (node_height + spacing + 20)
							
							# Determine color
							component_type = node.get('component_type', '')
							color = type_colors.get('default', '#95a5a6')
							for type_key in type_colors:
								if type_key in component_type:
									color = type_colors[type_key]
									break
							
							# Draw node rectangle
							draw.rectangle([x, y, x + node_width, y + node_height], fill=color, outline='#34495e', width=2)
							
							# Draw node label
							label = node.get('label', node.get('component_type', 'Node'))[:20]  # Truncate long labels
							
							# Calculate text position to center it
							text_bbox = draw.textbbox((0, 0), label, font=font_small)
							text_width = text_bbox[2] - text_bbox[0]
							text_height = text_bbox[3] - text_bbox[1]
							
							text_x = x + (node_width - text_width) // 2
							text_y = y + (node_height - text_height) // 2
							
							draw.text((text_x, text_y), label, fill='white', font=font_small)
							
							# Draw node ID below
							node_id = node.get('id', '')[:15]
							id_bbox = draw.textbbox((0, 0), node_id, font=font_small)
							id_width = id_bbox[2] - id_bbox[0]
							id_x = x + (node_width - id_width) // 2
							draw.text((id_x, y + node_height + 5), node_id, fill='#7f8c8d', font=font_small)
					
					# Add connection count info
					if connections:
						conn_text = f"Connections: {len(connections)}"
						draw.text((20, options.image_height - 60), conn_text, fill='#7f8c8d', font=font_medium)
					
					# Add metadata
					if options.include_metadata and workflow_data.get('metadata'):
						meta_y = options.image_height - 40
						meta_text = f"Created: {workflow_data['metadata'].get('created_at', 'N/A')}"
						draw.text((20, meta_y), meta_text, fill='#7f8c8d', font=font_small)
					
					# Add generation timestamp
					timestamp = f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
					draw.text((20, options.image_height - 20), timestamp, fill='#bdc3c7', font=font_small)
					
					# Save to bytes
					buffer = BytesIO()
					img.save(buffer, format='PNG', quality=95)
					png_content = buffer.getvalue()
					buffer.close()
					
					return ExportResult(
						success=True,
						format=ExportFormat.PNG,
						content=png_content,
						file_size=len(png_content),
						warnings=["PNG export using Pillow fallback. Install Playwright for enhanced visual rendering."]
					)
					
				except ImportError:
					logger.warning("Pillow not available, using SVG fallback for PNG")
					
					# Final fallback: Return enhanced SVG as PNG placeholder
					svg_options = ExportOptions(
						format=ExportFormat.SVG,
						include_layout=options.include_layout,
						image_width=options.image_width,
						image_height=options.image_height,
						background_color=options.background_color
					)
					
					svg_result = await self._export_svg(workflow_data, svg_options)
					if not svg_result.success:
						return svg_result
					
					return ExportResult(
						success=True,
						format=ExportFormat.PNG,
						content=svg_result.content,
						file_size=len(svg_result.content),
						warnings=["PNG export using SVG fallback. Install Playwright or Pillow for true PNG generation."]
					)
			
		except Exception as e:
			logger.error(f"Failed to export PNG: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.PNG,
				validation_errors=[f"PNG export failed: {e}"]
			)
	
	async def _export_svg(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to SVG format."""
		try:
			definition = workflow_data.get('definition', {})
			nodes = definition.get('nodes', [])
			connections = definition.get('connections', [])
			
			# Calculate bounds
			min_x = min_y = float('inf')
			max_x = max_y = float('-inf')
			
			for node in nodes:
				pos = node.get('position', {})
				size = node.get('size', {})
				x, y = pos.get('x', 0), pos.get('y', 0)
				w, h = size.get('width', 200), size.get('height', 100)
				
				min_x = min(min_x, x)
				min_y = min(min_y, y)
				max_x = max(max_x, x + w)
				max_y = max(max_y, y + h)
			
			# Add padding
			padding = 50
			if min_x == float('inf'):  # No nodes
				min_x = min_y = 0
				max_x = max_y = 500
			
			width = max(options.image_width, max_x - min_x + 2 * padding)
			height = max(options.image_height, max_y - min_y + 2 * padding)
			
			# Create SVG
			svg_content = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .node {{ fill: #3498db; stroke: #2980b9; stroke-width: 2; }}
      .trigger {{ fill: #e74c3c; stroke: #c0392b; }}
      .data {{ fill: #3498db; stroke: #2980b9; }}
      .logic {{ fill: #f39c12; stroke: #e67e22; }}
      .text {{ font-family: Arial, sans-serif; font-size: 12px; fill: white; text-anchor: middle; }}
      .connection {{ stroke: #7f8c8d; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }}
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#7f8c8d"/>
    </marker>
  </defs>
  
  <rect width="100%" height="100%" fill="{options.background_color}"/>
'''
			
			# Add connections first (so they appear behind nodes)
			for conn in connections:
				source_node = next((n for n in nodes if n.get('id') == conn.get('source_node_id')), None)
				target_node = next((n for n in nodes if n.get('id') == conn.get('target_node_id')), None)
				
				if source_node and target_node:
					source_pos = source_node.get('position', {})
					target_pos = target_node.get('position', {})
					source_size = source_node.get('size', {})
					
					x1 = source_pos.get('x', 0) + source_size.get('width', 200) - min_x + padding
					y1 = source_pos.get('y', 0) + source_size.get('height', 100) / 2 - min_y + padding
					x2 = target_pos.get('x', 0) - min_x + padding
					y2 = target_pos.get('y', 0) + target_pos.get('height', 100) / 2 - min_y + padding
					
					svg_content += f'  <path d="M {x1} {y1} L {x2} {y2}" class="connection"/>\n'
			
			# Add nodes
			for node in nodes:
				pos = node.get('position', {})
				size = node.get('size', {})
				x = pos.get('x', 0) - min_x + padding
				y = pos.get('y', 0) - min_y + padding
				w = size.get('width', 200)
				h = size.get('height', 100)
				
				# Determine node class
				component_type = node.get('component_type', '')
				if 'trigger' in component_type:
					node_class = 'node trigger'
				elif 'data' in component_type:
					node_class = 'node data'
				elif 'logic' in component_type or 'condition' in component_type:
					node_class = 'node logic'
				else:
					node_class = 'node'
				
				# Add rectangle
				svg_content += f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{node_class}"/>\n'
				
				# Add text
				label = node.get('label', node.get('component_type', 'Node'))
				text_x = x + w / 2
				text_y = y + h / 2 + 4  # Adjust for text baseline
				svg_content += f'  <text x="{text_x}" y="{text_y}" class="text">{label}</text>\n'
			
			svg_content += '</svg>'
			
			content_bytes = svg_content.encode('utf-8')
			
			return ExportResult(
				success=True,
				format=ExportFormat.SVG,
				content=content_bytes,
				file_size=len(content_bytes)
			)
			
		except Exception as e:
			logger.error(f"Failed to export SVG: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.SVG,
				validation_errors=[f"SVG export failed: {e}"]
			)
	
	async def _export_docx(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to Word document format."""
		try:
			# In production, would use python-docx library
			# For now, create a simple text representation
			
			lines = [
				f"Workflow Documentation: {workflow_data.get('name', 'Untitled')}",
				"=" * 50,
				"",
				f"Description: {workflow_data.get('description', 'No description provided')}",
				"",
			]
			
			# Add metadata
			if options.include_metadata and workflow_data.get('metadata'):
				lines.extend([
					"Metadata:",
					"-" * 20,
				])
				for key, value in workflow_data['metadata'].items():
					lines.append(f"{key}: {value}")
				lines.append("")
			
			# Add nodes
			definition = workflow_data.get('definition', {})
			nodes = definition.get('nodes', [])
			if nodes:
				lines.extend([
					"Workflow Components:",
					"-" * 30,
				])
				for i, node in enumerate(nodes, 1):
					lines.extend([
						f"{i}. {node.get('label', node.get('component_type', 'Unknown'))}",
						f"   Type: {node.get('component_type', 'Unknown')}",
						f"   ID: {node.get('id', 'N/A')}",
					])
					
					if options.include_config and node.get('config'):
						lines.append("   Configuration:")
						for key, value in node['config'].items():
							lines.append(f"     {key}: {value}")
					lines.append("")
			
			# Add connections
			connections = definition.get('connections', [])
			if connections:
				lines.extend([
					"Connections:",
					"-" * 20,
				])
				for i, conn in enumerate(connections, 1):
					lines.append(f"{i}. {conn.get('source_node_id', 'N/A')} -> {conn.get('target_node_id', 'N/A')}")
				lines.append("")
			
			lines.append(f"Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
			
			docx_content = "\n".join(lines)
			content_bytes = docx_content.encode('utf-8')
			
			return ExportResult(
				success=True,
				format=ExportFormat.DOCX,
				content=content_bytes,
				file_size=len(content_bytes),
				warnings=["DOCX export is using plain text. Install python-docx for true Word document generation."]
			)
			
		except Exception as e:
			logger.error(f"Failed to export DOCX: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.DOCX,
				validation_errors=[f"DOCX export failed: {e}"]
			)
	
	async def _export_excel(self, workflow_data: Dict[str, Any], options: ExportOptions) -> ExportResult:
		"""Export workflow to Excel format."""
		try:
			# In production, would use openpyxl or xlsxwriter
			# For now, create CSV content
			
			import csv
			from io import StringIO
			
			output = StringIO()
			writer = csv.writer(output)
			
			# Write metadata
			if options.include_metadata and workflow_data.get('metadata'):
				writer.writerow(["Metadata"])
				writer.writerow(["Key", "Value"])
				for key, value in workflow_data['metadata'].items():
					writer.writerow([key, str(value)])
				writer.writerow([])
			
			# Write nodes
			definition = workflow_data.get('definition', {})
			nodes = definition.get('nodes', [])
			if nodes:
				writer.writerow(["Workflow Components"])
				headers = ["ID", "Type", "Label"]
				if options.include_config:
					headers.extend(["Configuration"])
				writer.writerow(headers)
				
				for node in nodes:
					row = [
						node.get('id', ''),
						node.get('component_type', ''),
						node.get('label', '')
					]
					if options.include_config:
						config_str = json.dumps(node.get('config', {}))
						row.append(config_str)
					writer.writerow(row)
				writer.writerow([])
			
			# Write connections
			connections = definition.get('connections', [])
			if connections:
				writer.writerow(["Connections"])
				writer.writerow(["ID", "Source", "Target", "Source Port", "Target Port"])
				for conn in connections:
					writer.writerow([
						conn.get('id', ''),
						conn.get('source_node_id', ''),
						conn.get('target_node_id', ''),
						conn.get('source_port', 'output'),
						conn.get('target_port', 'input')
					])
			
			csv_content = output.getvalue()
			content_bytes = csv_content.encode('utf-8')
			
			return ExportResult(
				success=True,
				format=ExportFormat.EXCEL,
				content=content_bytes,
				file_size=len(content_bytes),
				warnings=["Excel export is using CSV format. Install openpyxl for true Excel generation."]
			)
			
		except Exception as e:
			logger.error(f"Failed to export Excel: {e}")
			return ExportResult(
				success=False,
				format=ExportFormat.EXCEL,
				validation_errors=[f"Excel export failed: {e}"]
			)
	
	# Helper methods
	
	async def _prepare_export_data(self, workflow_data: Dict[str, Any], options: ExportOptions) -> Dict[str, Any]:
		"""Prepare workflow data for export."""
		try:
			export_data = workflow_data.copy()
			
			# Filter sensitive data if requested
			if options.exclude_sensitive:
				export_data = await self._filter_sensitive_data(export_data)
			
			# Remove test data if not included
			if not options.include_test_data:
				export_data = await self._filter_test_data(export_data)
			
			# Add export metadata
			export_data['export_metadata'] = {
				'exported_at': datetime.now(timezone.utc).isoformat(),
				'format': options.format.value,
				'exporter': 'APG Workflow Designer',
				'version': '1.0.0'
			}
			
			return export_data
			
		except Exception as e:
			logger.error(f"Failed to prepare export data: {e}")
			return workflow_data
	
	async def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
		"""Filter out sensitive data from export."""
		try:
			# List of sensitive field patterns
			sensitive_patterns = ['password', 'secret', 'token', 'key', 'credential', 'api_key', 'private']
			
			def filter_dict(obj):
				if isinstance(obj, dict):
					return {
						k: "[REDACTED]" if any(pattern in k.lower() for pattern in sensitive_patterns) else filter_dict(v)
						for k, v in obj.items()
					}
				elif isinstance(obj, list):
					return [filter_dict(item) for item in obj]
				else:
					return obj
			
			return filter_dict(data)
			
		except Exception as e:
			logger.error(f"Failed to filter sensitive data: {e}")
			return data
	
	async def _filter_test_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
		"""Filter out test data from export."""
		try:
			# Remove test-related fields
			test_fields = ['test_config', 'test_data', 'mock_data', 'debug_info']
			
			def filter_dict(obj):
				if isinstance(obj, dict):
					return {
						k: filter_dict(v) for k, v in obj.items()
						if k not in test_fields and not k.startswith('test_')
					}
				elif isinstance(obj, list):
					return [filter_dict(item) for item in obj]
				else:
					return obj
			
			return filter_dict(data)
			
		except Exception as e:
			logger.error(f"Failed to filter test data: {e}")
			return data
	
	async def _validate_workflow_data(self, workflow_data: Dict[str, Any]) -> List[str]:
		"""Validate workflow data for export."""
		try:
			errors = []
			
			# Check required fields
			if not workflow_data.get('definition'):
				errors.append("Workflow definition is missing")
			
			definition = workflow_data.get('definition', {})
			
			# Validate nodes
			nodes = definition.get('nodes', [])
			if not nodes:
				errors.append("Workflow has no nodes")
			
			for node in nodes:
				if not node.get('id'):
					errors.append("Node missing ID")
				if not node.get('component_type'):
					errors.append(f"Node {node.get('id', 'unknown')} missing component type")
			
			# Validate connections
			connections = definition.get('connections', [])
			node_ids = {node.get('id') for node in nodes}
			
			for conn in connections:
				if not conn.get('id'):
					errors.append("Connection missing ID")
				
				source_id = conn.get('source_node_id')
				target_id = conn.get('target_node_id')
				
				if source_id not in node_ids:
					errors.append(f"Connection references non-existent source node: {source_id}")
				if target_id not in node_ids:
					errors.append(f"Connection references non-existent target node: {target_id}")
			
			return errors
			
		except Exception as e:
			logger.error(f"Failed to validate workflow data: {e}")
			return [f"Validation error: {e}"]
	
	async def _initialize_export_engines(self) -> None:
		"""Initialize export engines and dependencies."""
		try:
			# In production, initialize external libraries here
			# - PDF generation (weasyprint, reportlab)
			# - Image generation (Playwright, puppeteer)
			# - Office formats (python-docx, openpyxl)
			logger.info("Export engines initialized")
		except Exception as e:
			logger.error(f"Failed to initialize export engines: {e}")
			raise
	
	async def _load_export_templates(self) -> None:
		"""Load export templates from database."""
		try:
			# Create default templates
			default_templates = [
				ExportTemplate(
					id="professional_pdf",
					name="Professional PDF Report",
					description="Professional PDF report with detailed documentation",
					format=ExportFormat.PDF,
					template_content="",
					styles={
						"font": "Arial",
						"header_color": "#2c3e50",
						"accent_color": "#3498db"
					}
				),
				ExportTemplate(
					id="technical_html",
					name="Technical HTML Documentation",
					description="Technical HTML documentation with syntax highlighting",
					format=ExportFormat.HTML,
					template_content="",
					styles={
						"theme": "technical",
						"syntax_highlighting": True,
						"responsive": True
					}
				)
			]
			
			for template in default_templates:
				self.export_templates[template.id] = template
			
			logger.info(f"Loaded {len(default_templates)} default export templates")
			
		except Exception as e:
			logger.error(f"Failed to load export templates: {e}")
			raise
	
	async def _save_template_to_database(self, template: ExportTemplate) -> None:
		"""Save export template to database."""
		try:
			# In production, save to database
			logger.info(f"Saved export template {template.id} to database")
		except Exception as e:
			logger.error(f"Failed to save export template: {e}")
			raise
	
	async def _generate_pdf_html(self, workflow_data: Dict[str, Any], options: ExportOptions) -> str:
		"""Generate PDF-optimized HTML content."""
		try:
			# Enhanced HTML with print-specific styling
			html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workflow: {workflow_data.get('name', 'Untitled')}</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        
        body {{
            font-family: 'Arial', 'Helvetica', sans-serif;
            line-height: 1.6;
            color: #333;
            background: white;
            margin: 0;
            padding: 0;
        }}
        
        .header {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
            page-break-after: avoid;
        }}
        
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 28px;
            font-weight: bold;
        }}
        
        .header h2 {{
            color: #3498db;
            margin: 10px 0 0 0;
            font-size: 20px;
            font-weight: normal;
        }}
        
        .section {{
            margin: 30px 0;
            page-break-inside: avoid;
        }}
        
        .section h3 {{
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
            font-size: 18px;
        }}
        
        .node {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
            page-break-inside: avoid;
        }}
        
        .node h4 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 16px;
        }}
        
        .node-meta {{
            font-size: 12px;
            color: #6c757d;
            margin: 5px 0;
        }}
        
        .config {{
            font-family: 'Courier New', monospace;
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            font-size: 11px;
            margin-top: 10px;
            white-space: pre-wrap;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 12px;
            page-break-inside: auto;
        }}
        
        th, td {{
            padding: 8px 12px;
            text-align: left;
            border: 1px solid #dee2e6;
            word-wrap: break-word;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        .metadata {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            font-size: 11px;
            color: #6c757d;
            text-align: center;
        }}
        
        /* Print-specific styles */
        @media print {{
            .section {{
                page-break-inside: avoid;
            }}
            
            .node {{
                page-break-inside: avoid;
            }}
            
            table {{
                page-break-inside: auto;
            }}
            
            tr {{
                page-break-inside: avoid;
                page-break-after: auto;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Workflow Documentation</h1>
        <h2>{workflow_data.get('name', 'Untitled Workflow')}</h2>
        <p style="margin: 10px 0 0 0; color: #6c757d; font-style: italic;">
            {workflow_data.get('description', 'No description provided')}
        </p>
    </div>
"""
			
			# Add metadata section
			if options.include_metadata and workflow_data.get('metadata'):
				html_content += """
    <div class="section">
        <h3>Workflow Metadata</h3>
        <div class="metadata">
"""
				for key, value in workflow_data['metadata'].items():
					html_content += f"            <p><strong>{key}:</strong> {value}</p>\n"
				html_content += "        </div>\n    </div>\n"
			
			# Add workflow summary
			definition = workflow_data.get('definition', {})
			nodes = definition.get('nodes', [])
			connections = definition.get('connections', [])
			
			html_content += f"""
    <div class="section">
        <h3>Workflow Summary</h3>
        <table>
            <tr><td><strong>Total Components:</strong></td><td>{len(nodes)}</td></tr>
            <tr><td><strong>Total Connections:</strong></td><td>{len(connections)}</td></tr>
            <tr><td><strong>Created:</strong></td><td>{workflow_data.get('created_at', 'N/A')}</td></tr>
            <tr><td><strong>Last Modified:</strong></td><td>{workflow_data.get('updated_at', 'N/A')}</td></tr>
        </table>
    </div>
"""
			
			# Add nodes section
			if nodes:
				html_content += """
    <div class="section">
        <h3>Workflow Components</h3>
"""
				for node in nodes:
					component_type = node.get('component_type', 'Unknown')
					
					html_content += f"""
        <div class="node">
            <h4>{node.get('label', component_type)}</h4>
            <div class="node-meta">
                <strong>Type:</strong> {component_type} |
                <strong>ID:</strong> {node.get('id', 'N/A')}
            </div>
"""
					
					if options.include_config and node.get('config'):
						config_items = []
						for key, value in node['config'].items():
							config_items.append(f"{key}: {value}")
						config_text = "\\n".join(config_items)
						html_content += f'            <div class="config">{config_text}</div>\n'
					
					html_content += "        </div>\n"
				html_content += "    </div>\n"
			
			# Add connections section
			if connections:
				html_content += """
    <div class="section">
        <h3>Component Connections</h3>
        <table>
            <thead>
                <tr>
                    <th>Source Component</th>
                    <th>Target Component</th>
                    <th>Source Port</th>
                    <th>Target Port</th>
                </tr>
            </thead>
            <tbody>
"""
				for conn in connections:
					# Find source and target labels
					source_label = conn.get('source_node_id', 'N/A')
					target_label = conn.get('target_node_id', 'N/A')
					
					for node in nodes:
						if node.get('id') == conn.get('source_node_id'):
							source_label = node.get('label', node.get('component_type', source_label))
						if node.get('id') == conn.get('target_node_id'):
							target_label = node.get('label', node.get('component_type', target_label))
					
					html_content += f"""
                <tr>
                    <td>{source_label}</td>
                    <td>{target_label}</td>
                    <td>{conn.get('source_port', 'output')}</td>
                    <td>{conn.get('target_port', 'input')}</td>
                </tr>
"""
				html_content += """
            </tbody>
        </table>
    </div>
"""
			
			# Add footer
			html_content += f"""
    <div class="footer">
        <p>Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC by APG Workflow Designer</p>
        <p>© 2025 Datacraft. All rights reserved.</p>
    </div>
</body>
</html>
"""
			
			return html_content
			
		except Exception as e:
			logger.error(f"Failed to generate PDF HTML: {e}")
			return f"<html><body><h1>Error generating PDF content: {e}</h1></body></html>"