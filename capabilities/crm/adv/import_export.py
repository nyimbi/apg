"""
APG Customer Relationship Management - Import/Export Module

Advanced contact import/export functionality with support for multiple formats,
data validation, duplicate detection, and bulk operations.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import csv
import json
import logging
from datetime import datetime
from decimal import Decimal
from io import StringIO, BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
import pandas as pd
from uuid_extensions import uuid7str

from .models import CRMContact, ContactType, LeadSource
from .database import DatabaseManager


logger = logging.getLogger(__name__)


class ImportExportError(Exception):
	"""Base exception for import/export operations"""
	pass


class ValidationError(ImportExportError):
	"""Data validation error during import/export"""
	pass


class FormatError(ImportExportError):
	"""File format error during import/export"""
	pass


class ContactImportExportManager:
	"""
	Advanced contact import/export functionality supporting multiple formats
	and comprehensive data validation.
	"""
	
	SUPPORTED_IMPORT_FORMATS = ['csv', 'json', 'xlsx', 'vcf']
	SUPPORTED_EXPORT_FORMATS = ['csv', 'json', 'xlsx', 'pdf']
	
	# Standard field mappings for common CRM systems
	FIELD_MAPPINGS = {
		'salesforce': {
			'FirstName': 'first_name',
			'LastName': 'last_name',
			'Email': 'email',
			'Phone': 'phone',
			'Account.Name': 'company',
			'Title': 'job_title',
			'LeadSource': 'lead_source',
			'Description': 'description'
		},
		'hubspot': {
			'firstname': 'first_name',
			'lastname': 'last_name',
			'email': 'email',
			'phone': 'phone',
			'company': 'company',
			'jobtitle': 'job_title',
			'hs_lead_source': 'lead_source',
			'notes_last_contacted': 'notes'
		},
		'dynamics': {
			'firstname': 'first_name',
			'lastname': 'last_name',
			'emailaddress1': 'email',
			'telephone1': 'phone',
			'parentcustomerid_name': 'company',
			'jobtitle': 'job_title',
			'leadsourcecode': 'lead_source',
			'description': 'description'
		}
	}
	
	def __init__(self, database_manager: DatabaseManager, tenant_id: str):
		"""
		Initialize import/export manager
		
		Args:
			database_manager: Database manager instance
			tenant_id: Tenant identifier for multi-tenant isolation
		"""
		self.db_manager = database_manager
		self.tenant_id = tenant_id
		self.import_stats = {}
		self.export_stats = {}
	
	# ================================
	# Import Operations
	# ================================
	
	async def import_contacts(
		self,
		file_data: Union[str, bytes, BinaryIO],
		file_format: str,
		mapping_config: Optional[Dict[str, str]] = None,
		deduplicate: bool = True,
		validate_data: bool = True,
		created_by: str = "system"
	) -> Dict[str, Any]:
		"""
		Import contacts from various file formats
		
		Args:
			file_data: File content or file object
			file_format: Format type (csv, json, xlsx, vcf)
			mapping_config: Custom field mapping configuration
			deduplicate: Whether to check for duplicates
			validate_data: Whether to validate data before import
			created_by: User performing the import
			
		Returns:
			Import results with statistics and errors
		"""
		try:
			logger.info(f"🔄 Starting contact import - Format: {file_format}")
			
			if file_format not in self.SUPPORTED_IMPORT_FORMATS:
				raise FormatError(f"Unsupported import format: {file_format}")
			
			# Parse file data based on format
			raw_contacts = await self._parse_import_file(file_data, file_format)
			
			# Apply field mappings
			if mapping_config:
				mapped_contacts = self._apply_field_mapping(raw_contacts, mapping_config)
			else:
				mapped_contacts = raw_contacts
			
			# Validate contact data
			if validate_data:
				validated_contacts, validation_errors = await self._validate_import_data(mapped_contacts)
			else:
				validated_contacts = mapped_contacts
				validation_errors = []
			
			# Check for duplicates
			if deduplicate:
				unique_contacts, duplicate_info = await self._deduplicate_contacts(validated_contacts)
			else:
				unique_contacts = validated_contacts
				duplicate_info = []
			
			# Bulk import contacts
			import_results = await self._bulk_import_contacts(unique_contacts, created_by)
			
			# Compile import statistics
			stats = {
				"total_records": len(raw_contacts),
				"valid_records": len(validated_contacts),
				"imported_records": import_results["success_count"],
				"failed_records": import_results["error_count"],
				"duplicate_records": len(duplicate_info),
				"validation_errors": len(validation_errors),
				"processing_time": datetime.utcnow().isoformat(),
				"errors": import_results["errors"] + validation_errors,
				"duplicates": duplicate_info
			}
			
			logger.info(f"✅ Contact import completed - Imported: {stats['imported_records']}")
			return stats
			
		except Exception as e:
			logger.error(f"Contact import failed: {str(e)}", exc_info=True)
			raise ImportExportError(f"Import failed: {str(e)}")
	
	async def _parse_import_file(self, file_data: Union[str, bytes, BinaryIO], file_format: str) -> List[Dict[str, Any]]:
		"""Parse file data based on format"""
		try:
			if file_format == 'csv':
				return await self._parse_csv(file_data)
			elif file_format == 'json':
				return await self._parse_json(file_data)
			elif file_format == 'xlsx':
				return await self._parse_excel(file_data)
			elif file_format == 'vcf':
				return await self._parse_vcard(file_data)
			else:
				raise FormatError(f"Unsupported format: {file_format}")
		except Exception as e:
			raise FormatError(f"Failed to parse {file_format} file: {str(e)}")
	
	async def _parse_csv(self, file_data: Union[str, bytes]) -> List[Dict[str, Any]]:
		"""Parse CSV file data"""
		if isinstance(file_data, bytes):
			file_data = file_data.decode('utf-8')
		
		contacts = []
		csv_reader = csv.DictReader(StringIO(file_data))
		
		for row in csv_reader:
			# Clean and normalize data
			contact = {k.strip(): v.strip() if v else None for k, v in row.items()}
			contacts.append(contact)
		
		return contacts
	
	async def _parse_json(self, file_data: Union[str, bytes]) -> List[Dict[str, Any]]:
		"""Parse JSON file data"""
		if isinstance(file_data, bytes):
			file_data = file_data.decode('utf-8')
		
		data = json.loads(file_data)
		
		# Handle different JSON structures
		if isinstance(data, dict):
			# Single contact or wrapped data
			if 'contacts' in data:
				return data['contacts']
			else:
				return [data]
		elif isinstance(data, list):
			return data
		else:
			raise FormatError("Invalid JSON structure")
	
	async def _parse_excel(self, file_data: bytes) -> List[Dict[str, Any]]:
		"""Parse Excel file data"""
		try:
			df = pd.read_excel(BytesIO(file_data))
			# Convert DataFrame to list of dictionaries
			return df.to_dict('records')
		except Exception as e:
			raise FormatError(f"Failed to parse Excel file: {str(e)}")
	
	async def _parse_vcard(self, file_data: Union[str, bytes]) -> List[Dict[str, Any]]:
		"""Parse vCard file data"""
		if isinstance(file_data, bytes):
			file_data = file_data.decode('utf-8')
		
		contacts = []
		current_contact = {}
		
		for line in file_data.split('\n'):
			line = line.strip()
			
			if line.startswith('BEGIN:VCARD'):
				current_contact = {}
			elif line.startswith('END:VCARD'):
				if current_contact:
					contacts.append(current_contact)
			elif ':' in line:
				key, value = line.split(':', 1)
				
				# Map vCard fields to contact fields
				if key.startswith('FN'):
					# Try to split full name
					parts = value.split(' ', 1)
					current_contact['first_name'] = parts[0] if parts else value
					current_contact['last_name'] = parts[1] if len(parts) > 1 else ''
				elif key.startswith('EMAIL'):
					current_contact['email'] = value
				elif key.startswith('TEL'):
					current_contact['phone'] = value
				elif key.startswith('ORG'):
					current_contact['company'] = value
				elif key.startswith('TITLE'):
					current_contact['job_title'] = value
		
		return contacts
	
	def _apply_field_mapping(self, contacts: List[Dict[str, Any]], mapping: Dict[str, str]) -> List[Dict[str, Any]]:
		"""Apply field mapping to normalize contact data"""
		mapped_contacts = []
		
		for contact in contacts:
			mapped_contact = {}
			
			for source_field, target_field in mapping.items():
				if source_field in contact:
					mapped_contact[target_field] = contact[source_field]
			
			# Copy unmapped fields
			for field, value in contact.items():
				if field not in mapping and field not in mapped_contact:
					mapped_contact[field] = value
			
			mapped_contacts.append(mapped_contact)
		
		return mapped_contacts
	
	async def _validate_import_data(self, contacts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
		"""Validate contact data before import"""
		valid_contacts = []
		validation_errors = []
		
		for i, contact_data in enumerate(contacts):
			try:
				# Required fields validation
				if not contact_data.get('first_name') and not contact_data.get('last_name'):
					validation_errors.append({
						"row": i + 1,
						"error": "Missing required field: first_name or last_name",
						"data": contact_data
					})
					continue
				
				# Email validation
				email = contact_data.get('email')
				if email and '@' not in email:
					validation_errors.append({
						"row": i + 1,
						"error": f"Invalid email format: {email}",
						"data": contact_data
					})
					continue
				
				# Normalize data types
				normalized_contact = await self._normalize_contact_data(contact_data)
				valid_contacts.append(normalized_contact)
				
			except Exception as e:
				validation_errors.append({
					"row": i + 1,
					"error": f"Validation error: {str(e)}",
					"data": contact_data
				})
		
		return valid_contacts, validation_errors
	
	async def _normalize_contact_data(self, contact_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Normalize contact data to match CRMContact model"""
		normalized = {
			"id": uuid7str(),
			"tenant_id": self.tenant_id,
			"first_name": contact_data.get('first_name', '').strip(),
			"last_name": contact_data.get('last_name', '').strip(),
			"email": contact_data.get('email', '').strip().lower() if contact_data.get('email') else None,
			"phone": contact_data.get('phone', '').strip() if contact_data.get('phone') else None,
			"mobile": contact_data.get('mobile', '').strip() if contact_data.get('mobile') else None,
			"company": contact_data.get('company', '').strip() if contact_data.get('company') else None,
			"job_title": contact_data.get('job_title', '').strip() if contact_data.get('job_title') else None,
			"department": contact_data.get('department', '').strip() if contact_data.get('department') else None,
			"website": contact_data.get('website', '').strip() if contact_data.get('website') else None,
			"linkedin_profile": contact_data.get('linkedin_profile', '').strip() if contact_data.get('linkedin_profile') else None,
			"contact_type": ContactType.PROSPECT,  # Default type
			"lead_source": LeadSource.IMPORT if contact_data.get('lead_source') else None,
			"description": contact_data.get('description', '').strip() if contact_data.get('description') else None,
			"notes": contact_data.get('notes', '').strip() if contact_data.get('notes') else None,
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow(),
			"version": 1
		}
		
		# Handle lead score if present
		if 'lead_score' in contact_data:
			try:
				normalized['lead_score'] = float(contact_data['lead_score'])
			except (ValueError, TypeError):
				normalized['lead_score'] = None
		
		return normalized
	
	async def _deduplicate_contacts(self, contacts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
		"""Check for duplicate contacts and handle accordingly"""
		unique_contacts = []
		duplicate_info = []
		
		# Create lookup for existing contacts by email
		existing_emails = set()
		if contacts:
			emails = [c.get('email') for c in contacts if c.get('email')]
			if emails:
				existing_contacts = await self.db_manager.find_contacts_by_emails(self.tenant_id, emails)
				existing_emails = {c.email for c in existing_contacts if c.email}
		
		# Check for duplicates
		seen_emails = set()
		for contact in contacts:
			email = contact.get('email')
			
			if not email:
				unique_contacts.append(contact)
				continue
			
			# Check against existing database records
			if email in existing_emails:
				duplicate_info.append({
					"type": "existing_database",
					"email": email,
					"contact": contact
				})
				continue
			
			# Check against current import batch
			if email in seen_emails:
				duplicate_info.append({
					"type": "duplicate_in_batch",
					"email": email,
					"contact": contact
				})
				continue
			
			seen_emails.add(email)
			unique_contacts.append(contact)
		
		return unique_contacts, duplicate_info
	
	async def _bulk_import_contacts(self, contacts: List[Dict[str, Any]], created_by: str) -> Dict[str, Any]:
		"""Perform bulk import of validated contacts"""
		success_count = 0
		error_count = 0
		errors = []
		
		# Process in batches for better performance
		batch_size = 100
		for i in range(0, len(contacts), batch_size):
			batch = contacts[i:i + batch_size]
			
			try:
				# Add audit fields
				for contact in batch:
					contact['created_by'] = created_by
					contact['updated_by'] = created_by
				
				# Bulk insert batch
				batch_results = await self.db_manager.bulk_create_contacts(batch)
				success_count += batch_results['success_count']
				error_count += batch_results['error_count']
				errors.extend(batch_results['errors'])
				
			except Exception as e:
				error_count += len(batch)
				errors.append({
					"batch": f"{i}-{i+len(batch)}",
					"error": str(e)
				})
		
		return {
			"success_count": success_count,
			"error_count": error_count,
			"errors": errors
		}
	
	# ================================
	# Export Operations
	# ================================
	
	async def export_contacts(
		self,
		export_format: str,
		contact_ids: Optional[List[str]] = None,
		filters: Optional[Dict[str, Any]] = None,
		include_fields: Optional[List[str]] = None,
		exclude_fields: Optional[List[str]] = None
	) -> Tuple[Union[str, bytes], str]:
		"""
		Export contacts to various formats
		
		Args:
			export_format: Format for export (csv, json, xlsx, pdf)
			contact_ids: Specific contact IDs to export
			filters: Filters to apply to contact selection
			include_fields: Fields to include in export
			exclude_fields: Fields to exclude from export
			
		Returns:
			Tuple of (exported_data, filename)
		"""
		try:
			logger.info(f"🔄 Starting contact export - Format: {export_format}")
			
			if export_format not in self.SUPPORTED_EXPORT_FORMATS:
				raise FormatError(f"Unsupported export format: {export_format}")
			
			# Get contacts to export
			contacts = await self._get_contacts_for_export(contact_ids, filters)
			
			if not contacts:
				raise ImportExportError("No contacts found for export")
			
			# Filter fields if specified
			filtered_contacts = self._filter_export_fields(contacts, include_fields, exclude_fields)
			
			# Generate export data based on format
			if export_format == 'csv':
				export_data, filename = await self._export_csv(filtered_contacts)
			elif export_format == 'json':
				export_data, filename = await self._export_json(filtered_contacts)
			elif export_format == 'xlsx':
				export_data, filename = await self._export_excel(filtered_contacts)
			elif export_format == 'pdf':
				export_data, filename = await self._export_pdf(filtered_contacts)
			
			# Update export statistics
			self.export_stats = {
				"exported_records": len(contacts),
				"export_format": export_format,
				"export_time": datetime.utcnow().isoformat(),
				"filename": filename
			}
			
			logger.info(f"✅ Contact export completed - Records: {len(contacts)}")
			return export_data, filename
			
		except Exception as e:
			logger.error(f"Contact export failed: {str(e)}", exc_info=True)
			raise ImportExportError(f"Export failed: {str(e)}")
	
	async def _get_contacts_for_export(
		self,
		contact_ids: Optional[List[str]] = None,
		filters: Optional[Dict[str, Any]] = None
	) -> List[CRMContact]:
		"""Get contacts for export based on criteria"""
		if contact_ids:
			# Export specific contacts
			contacts = []
			for contact_id in contact_ids:
				contact = await self.db_manager.get_contact(contact_id, self.tenant_id)
				if contact:
					contacts.append(contact)
			return contacts
		else:
			# Export filtered contacts
			result = await self.db_manager.list_contacts(
				self.tenant_id,
				filters=filters,
				limit=10000  # Large limit for export
			)
			return result['items']
	
	def _filter_export_fields(
		self,
		contacts: List[CRMContact],
		include_fields: Optional[List[str]] = None,
		exclude_fields: Optional[List[str]] = None
	) -> List[Dict[str, Any]]:
		"""Filter contact fields for export"""
		filtered_contacts = []
		
		for contact in contacts:
			contact_dict = contact.model_dump()
			
			if include_fields:
				# Only include specified fields
				filtered_dict = {field: contact_dict.get(field) for field in include_fields}
			elif exclude_fields:
				# Exclude specified fields
				filtered_dict = {k: v for k, v in contact_dict.items() if k not in exclude_fields}
			else:
				# Include all fields
				filtered_dict = contact_dict
			
			# Convert datetime objects to strings
			for key, value in filtered_dict.items():
				if isinstance(value, datetime):
					filtered_dict[key] = value.isoformat()
				elif isinstance(value, Decimal):
					filtered_dict[key] = float(value)
			
			filtered_contacts.append(filtered_dict)
		
		return filtered_contacts
	
	async def _export_csv(self, contacts: List[Dict[str, Any]]) -> Tuple[str, str]:
		"""Export contacts to CSV format"""
		if not contacts:
			return "", "contacts_empty.csv"
		
		output = StringIO()
		
		# Get all field names
		fieldnames = set()
		for contact in contacts:
			fieldnames.update(contact.keys())
		fieldnames = sorted(list(fieldnames))
		
		writer = csv.DictWriter(output, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(contacts)
		
		csv_data = output.getvalue()
		filename = f"contacts_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
		
		return csv_data, filename
	
	async def _export_json(self, contacts: List[Dict[str, Any]]) -> Tuple[str, str]:
		"""Export contacts to JSON format"""
		export_data = {
			"contacts": contacts,
			"export_metadata": {
				"total_records": len(contacts),
				"export_time": datetime.utcnow().isoformat(),
				"tenant_id": self.tenant_id
			}
		}
		
		json_data = json.dumps(export_data, indent=2, default=str)
		filename = f"contacts_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
		
		return json_data, filename
	
	async def _export_excel(self, contacts: List[Dict[str, Any]]) -> Tuple[bytes, str]:
		"""Export contacts to Excel format"""
		try:
			df = pd.DataFrame(contacts)
			
			output = BytesIO()
			with pd.ExcelWriter(output, engine='openpyxl') as writer:
				df.to_excel(writer, sheet_name='Contacts', index=False)
			
			excel_data = output.getvalue()
			filename = f"contacts_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
			
			return excel_data, filename
			
		except Exception as e:
			raise FormatError(f"Failed to create Excel export: {str(e)}")
	
	async def _export_pdf(self, contacts: List[Dict[str, Any]]) -> Tuple[bytes, str]:
		"""Export contacts to PDF format"""
		try:
			from reportlab.lib import colors
			from reportlab.lib.pagesizes import letter, A4
			from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
			from reportlab.lib.styles import getSampleStyleSheet
			
			output = BytesIO()
			doc = SimpleDocTemplate(output, pagesize=A4)
			
			# Create content
			styles = getSampleStyleSheet()
			title = Paragraph("Contact Export Report", styles['Title'])
			
			# Create table data
			if contacts:
				# Limit fields for PDF readability
				pdf_fields = ['first_name', 'last_name', 'email', 'phone', 'company', 'job_title']
				
				# Filter contacts to only include PDF fields
				table_data = []
				headers = [field.replace('_', ' ').title() for field in pdf_fields]
				table_data.append(headers)
				
				for contact in contacts:
					row = [str(contact.get(field, '')) for field in pdf_fields]
					table_data.append(row)
				
				# Create table
				table = Table(table_data)
				table.setStyle(TableStyle([
					('BACKGROUND', (0, 0), (-1, 0), colors.grey),
					('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
					('ALIGN', (0, 0), (-1, -1), 'CENTER'),
					('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
					('FONTSIZE', (0, 0), (-1, 0), 12),
					('BOTTOMPADDING', (0, 0), (-1, 0), 12),
					('BACKGROUND', (0, 1), (-1, -1), colors.beige),
					('GRID', (0, 0), (-1, -1), 1, colors.black)
				]))
				
				story = [title, table]
			else:
				story = [title, Paragraph("No contacts found for export.", styles['Normal'])]
			
			doc.build(story)
			
			pdf_data = output.getvalue()
			filename = f"contacts_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
			
			return pdf_data, filename
			
		except ImportError:
			raise FormatError("PDF export requires reportlab package: pip install reportlab")
		except Exception as e:
			raise FormatError(f"Failed to create PDF export: {str(e)}")
	
	# ================================
	# Template Operations
	# ================================
	
	async def get_import_template(self, file_format: str, mapping_type: Optional[str] = None) -> Tuple[Union[str, bytes], str]:
		"""
		Generate import template for specified format
		
		Args:
			file_format: Template format (csv, json, xlsx)
			mapping_type: Predefined mapping type (salesforce, hubspot, dynamics)
			
		Returns:
			Tuple of (template_data, filename)
		"""
		try:
			# Define template fields
			if mapping_type and mapping_type in self.FIELD_MAPPINGS:
				# Use mapped field names
				template_fields = list(self.FIELD_MAPPINGS[mapping_type].keys())
			else:
				# Use standard CRM field names
				template_fields = [
					'first_name', 'last_name', 'email', 'phone', 'mobile',
					'company', 'job_title', 'department', 'website',
					'linkedin_profile', 'lead_source', 'description', 'notes'
				]
			
			if file_format == 'csv':
				return await self._generate_csv_template(template_fields)
			elif file_format == 'json':
				return await self._generate_json_template(template_fields)
			elif file_format == 'xlsx':
				return await self._generate_excel_template(template_fields)
			else:
				raise FormatError(f"Template not available for format: {file_format}")
				
		except Exception as e:
			logger.error(f"Template generation failed: {str(e)}", exc_info=True)
			raise ImportExportError(f"Template generation failed: {str(e)}")
	
	async def _generate_csv_template(self, fields: List[str]) -> Tuple[str, str]:
		"""Generate CSV import template"""
		output = StringIO()
		writer = csv.writer(output)
		writer.writerow(fields)
		
		# Add example row
		example_row = []
		for field in fields:
			if 'first_name' in field.lower():
				example_row.append('John')
			elif 'last_name' in field.lower():
				example_row.append('Doe')
			elif 'email' in field.lower():
				example_row.append('john.doe@example.com')
			elif 'phone' in field.lower():
				example_row.append('+1-555-123-4567')
			elif 'company' in field.lower():
				example_row.append('Example Corp')
			elif 'job_title' in field.lower() or 'title' in field.lower():
				example_row.append('Marketing Manager')
			else:
				example_row.append('')
		
		writer.writerow(example_row)
		
		template_data = output.getvalue()
		filename = "contact_import_template.csv"
		
		return template_data, filename
	
	async def _generate_json_template(self, fields: List[str]) -> Tuple[str, str]:
		"""Generate JSON import template"""
		example_contact = {}
		for field in fields:
			if 'first_name' in field.lower():
				example_contact[field] = 'John'
			elif 'last_name' in field.lower():
				example_contact[field] = 'Doe'
			elif 'email' in field.lower():
				example_contact[field] = 'john.doe@example.com'
			elif 'phone' in field.lower():
				example_contact[field] = '+1-555-123-4567'
			elif 'company' in field.lower():
				example_contact[field] = 'Example Corp'
			elif 'job_title' in field.lower() or 'title' in field.lower():
				example_contact[field] = 'Marketing Manager'
			else:
				example_contact[field] = ''
		
		template_data = {
			"contacts": [example_contact],
			"import_instructions": {
				"description": "Import template for CRM contacts",
				"fields": fields,
				"notes": "Replace example data with actual contact information"
			}
		}
		
		json_data = json.dumps(template_data, indent=2)
		filename = "contact_import_template.json"
		
		return json_data, filename
	
	async def _generate_excel_template(self, fields: List[str]) -> Tuple[bytes, str]:
		"""Generate Excel import template"""
		try:
			# Create example data
			example_data = {}
			for field in fields:
				if 'first_name' in field.lower():
					example_data[field] = 'John'
				elif 'last_name' in field.lower():
					example_data[field] = 'Doe'
				elif 'email' in field.lower():
					example_data[field] = 'john.doe@example.com'
				elif 'phone' in field.lower():
					example_data[field] = '+1-555-123-4567'
				elif 'company' in field.lower():
					example_data[field] = 'Example Corp'
				elif 'job_title' in field.lower() or 'title' in field.lower():
					example_data[field] = 'Marketing Manager'
				else:
					example_data[field] = ''
			
			df = pd.DataFrame([example_data])
			
			output = BytesIO()
			with pd.ExcelWriter(output, engine='openpyxl') as writer:
				df.to_excel(writer, sheet_name='Contact Import Template', index=False)
			
			template_data = output.getvalue()
			filename = "contact_import_template.xlsx"
			
			return template_data, filename
			
		except Exception as e:
			raise FormatError(f"Failed to create Excel template: {str(e)}")