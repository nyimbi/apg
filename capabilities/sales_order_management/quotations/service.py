"""
Quotations Service

Business logic for quotation management, quote-to-order conversion,
and quotation workflow processing.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from .models import SOQQuotation, SOQQuotationLine, SOQQuoteTemplate


class QuotationsService:
	"""Service class for quotation operations"""
	
	def __init__(self, db_session: Session):
		self.db = db_session
	
	def create_quotation(self, tenant_id: str, quote_data: Dict[str, Any], user_id: str) -> SOQQuotation:
		"""Create a new quotation"""
		quote_number = self._generate_quote_number(tenant_id)
		
		quotation = SOQQuotation(
			tenant_id=tenant_id,
			quote_number=quote_number,
			quote_name=quote_data['quote_name'],
			description=quote_data.get('description'),
			customer_id=quote_data['customer_id'],
			customer_name=quote_data['customer_name'],
			contact_name=quote_data.get('contact_name'),
			contact_email=quote_data.get('contact_email'),
			quote_date=quote_data.get('quote_date', date.today()),
			valid_until_date=quote_data.get('valid_until_date', date.today() + timedelta(days=30)),
			sales_rep_id=quote_data.get('sales_rep_id'),
			payment_terms=quote_data.get('payment_terms'),
			delivery_terms=quote_data.get('delivery_terms'),
			notes=quote_data.get('notes'),
			created_by_user_id=user_id
		)
		
		self.db.add(quotation)
		self.db.flush()
		
		# Add quotation lines
		if 'lines' in quote_data:
			for line_num, line_data in enumerate(quote_data['lines'], 1):
				self._create_quotation_line(quotation, line_num, line_data, user_id)
		
		# Calculate totals
		quotation.calculate_totals()
		
		self.db.commit()
		return quotation
	
	def convert_to_order(self, tenant_id: str, quotation_id: str, user_id: str, 
						conversion_data: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Convert quotation to sales order"""
		quotation = self.db.query(SOQQuotation).filter(
			and_(
				SOQQuotation.tenant_id == tenant_id,
				SOQQuotation.quotation_id == quotation_id
			)
		).first()
		
		if not quotation:
			raise ValueError("Quotation not found")
		
		if not quotation.is_valid():
			raise ValueError("Quotation is not valid for conversion")
		
		# Create order data from quotation
		order_data = {
			'customer_id': quotation.customer_id,
			'quote_id': quotation.quotation_id,
			'order_date': date.today(),
			'requested_date': conversion_data.get('requested_date') if conversion_data else None,
			'description': f"Order from Quote {quotation.quote_number}",
			'payment_terms_code': quotation.payment_terms,
			'sales_rep_id': quotation.sales_rep_id,
			'notes': quotation.notes,
			'lines': []
		}
		
		# Convert quotation lines to order lines
		for quote_line in quotation.lines:
			if quote_line.line_type in ['PRODUCT', 'SERVICE']:
				line_data = {
					'item_code': quote_line.item_code,
					'item_description': quote_line.item_description,
					'quantity_ordered': quote_line.quantity,
					'unit_price': quote_line.unit_price,
					'discount_percentage': quote_line.discount_percentage,
					'tax_code': quote_line.tax_code,
					'notes': quote_line.notes
				}
				order_data['lines'].append(line_data)
		
		# Mark quotation as converted (order_id would be set after order creation)
		quotation.customer_response = 'ACCEPTED'
		quotation.customer_response_date = date.today()
		quotation.status = 'CONVERTED'
		quotation.conversion_date = date.today()
		
		self.db.commit()
		
		return {
			'success': True,
			'order_data': order_data,
			'quotation': quotation
		}
	
	def create_revision(self, tenant_id: str, quotation_id: str, revision_data: Dict[str, Any], user_id: str) -> SOQQuotation:
		"""Create a revision of an existing quotation"""
		original_quote = self.db.query(SOQQuotation).filter(
			and_(
				SOQQuotation.tenant_id == tenant_id,
				SOQQuotation.quotation_id == quotation_id
			)
		).first()
		
		if not original_quote:
			raise ValueError("Original quotation not found")
		
		# Mark original as not current
		original_quote.is_current_revision = False
		
		# Create new revision
		new_quote_number = f"{original_quote.quote_number}-R{original_quote.revision_number + 1}"
		
		revision = SOQQuotation(
			tenant_id=tenant_id,
			quote_number=new_quote_number,
			quote_name=revision_data.get('quote_name', original_quote.quote_name),
			description=revision_data.get('description', original_quote.description),
			customer_id=original_quote.customer_id,
			customer_name=original_quote.customer_name,
			contact_name=revision_data.get('contact_name', original_quote.contact_name),
			contact_email=revision_data.get('contact_email', original_quote.contact_email),
			quote_date=date.today(),
			valid_until_date=revision_data.get('valid_until_date', date.today() + timedelta(days=30)),
			sales_rep_id=original_quote.sales_rep_id,
			revision_number=original_quote.revision_number + 1,
			parent_quote_id=original_quote.quotation_id,
			payment_terms=revision_data.get('payment_terms', original_quote.payment_terms),
			delivery_terms=revision_data.get('delivery_terms', original_quote.delivery_terms),
			notes=revision_data.get('notes', original_quote.notes),
			created_by_user_id=user_id
		)
		
		self.db.add(revision)
		self.db.flush()
		
		# Copy and update lines
		for orig_line in original_quote.lines:
			# Find corresponding line in revision data or copy original
			line_updates = {}
			if 'lines' in revision_data:
				for line_data in revision_data['lines']:
					if line_data.get('original_line_id') == orig_line.line_id:
						line_updates = line_data
						break
			
			self._create_quotation_line(
				revision, 
				orig_line.line_number, 
				{
					'item_code': line_updates.get('item_code', orig_line.item_code),
					'item_description': line_updates.get('item_description', orig_line.item_description),
					'quantity': line_updates.get('quantity', orig_line.quantity),
					'unit_price': line_updates.get('unit_price', orig_line.unit_price),
					'discount_percentage': line_updates.get('discount_percentage', orig_line.discount_percentage),
					'notes': line_updates.get('notes', orig_line.notes)
				},
				user_id
			)
		
		revision.calculate_totals()
		self.db.commit()
		
		return revision
	
	def _create_quotation_line(self, quotation: SOQQuotation, line_number: int, 
							  line_data: Dict[str, Any], user_id: str) -> SOQQuotationLine:
		"""Create a quotation line"""
		line = SOQQuotationLine(
			quotation_id=quotation.quotation_id,
			tenant_id=quotation.tenant_id,
			line_number=line_number,
			line_type=line_data.get('line_type', 'PRODUCT'),
			item_code=line_data.get('item_code'),
			item_description=line_data.get('item_description'),
			description=line_data.get('description'),
			quantity=line_data.get('quantity', 1),
			unit_of_measure=line_data.get('unit_of_measure', 'EA'),
			list_price=line_data.get('list_price', 0),
			unit_price=line_data.get('unit_price', 0),
			discount_percentage=line_data.get('discount_percentage', 0),
			tax_code=line_data.get('tax_code'),
			lead_time_days=line_data.get('lead_time_days'),
			notes=line_data.get('notes'),
			created_by_user_id=user_id
		)
		
		# Calculate amounts
		line.calculate_extended_amount()
		line.calculate_margin()
		
		# Calculate tax if applicable
		if line.tax_code and line.is_taxable:
			line.tax_rate = self._get_tax_rate(line.tax_code)
			line.tax_amount = line.extended_amount * (line.tax_rate / 100)
		
		self.db.add(line)
		return line
	
	def _generate_quote_number(self, tenant_id: str) -> str:
		"""Generate next quote number"""
		count = self.db.query(SOQQuotation).filter(SOQQuotation.tenant_id == tenant_id).count()
		return f"QT-{count + 1:06d}"
	
	def _get_tax_rate(self, tax_code: str) -> Decimal:
		"""Get tax rate for tax code"""
		# Mock implementation
		return Decimal('8.25')