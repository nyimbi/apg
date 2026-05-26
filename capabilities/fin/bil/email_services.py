"""
APG Billing Email Services

Real email delivery services for invoices, payment notifications, and billing communications
using SendGrid, Mailgun, Amazon SES, and other providers.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

try:
	import aiohttp
except ImportError:  # pragma: no cover - exercised through billing import regression
	aiohttp = None

try:
	import boto3
except ImportError:  # pragma: no cover - exercised through billing import regression
	boto3 = None

try:
	from sendgrid import SendGridAPIClient
	from sendgrid.helpers.mail import Mail, From, To, Subject, PlainTextContent, HtmlContent, Attachment, FileContent, FileName, FileType
except ImportError:  # pragma: no cover - exercised through billing import regression
	SendGridAPIClient = None
	Mail = From = To = Subject = PlainTextContent = HtmlContent = Attachment = FileContent = FileName = FileType = None

from .models import BLCustomer, BLInvoice, BLPayment


class EmailServiceError(Exception):
	"""Email service error"""
	pass


class EmailDeliveryError(Exception):
	"""Email delivery error"""
	pass


class EmailService(ABC):
	"""Abstract base class for email services"""
	
	@abstractmethod
	async def send_email(self, to_email: str, subject: str, html_content: str, 
						text_content: str = None, attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Send email with optional attachments"""
		pass
	
	@abstractmethod
	async def send_template_email(self, to_email: str, template_id: str, 
								 template_data: Dict[str, Any], attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Send email using template"""
		pass
	
	@abstractmethod
	async def validate_email(self, email: str) -> Dict[str, Any]:
		"""Validate email address"""
		pass


class SendGridEmailService(EmailService):
	"""SendGrid email service implementation"""
	
	def __init__(self, api_key: str, from_email: str, from_name: str = None):
		if SendGridAPIClient is None:
			raise EmailServiceError("SendGrid SDK is required to initialize SendGrid email service")
		self.api_key = api_key
		self.from_email = from_email
		self.from_name = from_name or "APG Billing"
		self.client = SendGridAPIClient(api_key=api_key)
		self.logger = logging.getLogger(f"{__name__}.SendGridEmailService")
	
	async def send_email(self, to_email: str, subject: str, html_content: str, 
						text_content: str = None, attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Send email via SendGrid"""
		try:
			# Create message
			message = Mail()
			message.from_email = From(self.from_email, self.from_name)
			message.to = To(to_email)
			message.subject = Subject(subject)
			
			# Add content
			if text_content:
				message.content = PlainTextContent(text_content)
			message.content = HtmlContent(html_content)
			
			# Add attachments
			if attachments:
				for attachment in attachments:
					sg_attachment = Attachment()
					sg_attachment.file_content = FileContent(attachment['content'])
					sg_attachment.file_name = FileName(attachment['filename'])
					sg_attachment.file_type = FileType(attachment.get('content_type', 'application/pdf'))
					message.attachment = sg_attachment
			
			# Send email
			response = self.client.send(message)
			
			return {
				'success': True,
				'message_id': response.headers.get('X-Message-Id'),
				'status_code': response.status_code,
				'provider': 'sendgrid'
			}
		
		except Exception as e:
			self.logger.error(f"SendGrid email failed: {e}")
			raise EmailDeliveryError(f"SendGrid email failed: {e}")
	
	async def send_template_email(self, to_email: str, template_id: str, 
								 template_data: Dict[str, Any], attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Send templated email via SendGrid"""
		try:
			message = Mail()
			message.from_email = From(self.from_email, self.from_name)
			message.to = To(to_email)
			message.template_id = template_id
			
			# Add dynamic template data
			message.dynamic_template_data = template_data
			
			# Add attachments
			if attachments:
				for attachment in attachments:
					sg_attachment = Attachment()
					sg_attachment.file_content = FileContent(attachment['content'])
					sg_attachment.file_name = FileName(attachment['filename'])
					sg_attachment.file_type = FileType(attachment.get('content_type', 'application/pdf'))
					message.attachment = sg_attachment
			
			response = self.client.send(message)
			
			return {
				'success': True,
				'message_id': response.headers.get('X-Message-Id'),
				'status_code': response.status_code,
				'provider': 'sendgrid'
			}
		
		except Exception as e:
			self.logger.error(f"SendGrid template email failed: {e}")
			raise EmailDeliveryError(f"SendGrid template email failed: {e}")
	
	async def validate_email(self, email: str) -> Dict[str, Any]:
		"""Validate email using SendGrid validation API"""
		try:
			# Use SendGrid email validation
			validation_response = self.client.validations.email_validation.post(
				request_body={'email': email}
			)
			
			return {
				'is_valid': validation_response.body.get('verdict') == 'Valid',
				'email': email,
				'provider': 'sendgrid',
				'details': validation_response.body
			}
		
		except Exception as e:
			self.logger.warning(f"Email validation failed: {e}")
			return {
				'is_valid': '@' in email and '.' in email,  # Basic fallback
				'email': email,
				'error': str(e)
			}


class MailgunEmailService(EmailService):
	"""Mailgun email service implementation"""
	
	def __init__(self, api_key: str, domain: str, from_email: str, from_name: str = None):
		if aiohttp is None:
			raise EmailServiceError("aiohttp is required to initialize Mailgun email service")
		self.api_key = api_key
		self.domain = domain
		self.from_email = from_email
		self.from_name = from_name or "APG Billing"
		self.base_url = f"https://api.mailgun.net/v3/{domain}"
		self.logger = logging.getLogger(f"{__name__}.MailgunEmailService")
	
	async def send_email(self, to_email: str, subject: str, html_content: str, 
						text_content: str = None, attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Send email via Mailgun"""
		try:
			async with aiohttp.ClientSession() as session:
				# Prepare form data
				data = {
					'from': f"{self.from_name} <{self.from_email}>",
					'to': to_email,
					'subject': subject,
					'html': html_content
				}
				
				if text_content:
					data['text'] = text_content
				
				# Add attachments for Mailgun
				if attachments:
					# Mailgun handles attachments as multipart form data
					# Convert data to FormData for multipart upload
					form_data = aiohttp.FormData()
					for key, value in data.items():
						form_data.add_field(key, value)
					
					# Add attachment files
					for attachment in attachments:
						if 'content' in attachment and 'filename' in attachment:
							content_type = attachment.get('content_type', 'application/octet-stream')
							form_data.add_field(
								'attachment',
								attachment['content'],
								filename=attachment['filename'],
								content_type=content_type
							)
					data = form_data
				else:
					# Keep as regular dict for non-multipart requests
					pass
				
				# Send email
				auth = aiohttp.BasicAuth('api', self.api_key)
				async with session.post(
					f"{self.base_url}/messages",
					auth=auth,
					data=data
				) as response:
					result = await response.json()
					
					if response.status == 200:
						return {
							'success': True,
							'message_id': result.get('id'),
							'message': result.get('message'),
							'provider': 'mailgun'
						}
					else:
						raise EmailDeliveryError(f"Mailgun error: {result}")
		
		except Exception as e:
			self.logger.error(f"Mailgun email failed: {e}")
			raise EmailDeliveryError(f"Mailgun email failed: {e}")
	
	async def send_template_email(self, to_email: str, template_id: str, 
								 template_data: Dict[str, Any], attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Send templated email via Mailgun"""
		try:
			async with aiohttp.ClientSession() as session:
				data = {
					'from': f"{self.from_name} <{self.from_email}>",
					'to': to_email,
					'template': template_id,
					'h:X-Mailgun-Variables': json.dumps(template_data)
				}
				
				auth = aiohttp.BasicAuth('api', self.api_key)
				async with session.post(
					f"{self.base_url}/messages",
					auth=auth,
					data=data
				) as response:
					result = await response.json()
					
					if response.status == 200:
						return {
							'success': True,
							'message_id': result.get('id'),
							'provider': 'mailgun'
						}
					else:
						raise EmailDeliveryError(f"Mailgun template error: {result}")
		
		except Exception as e:
			self.logger.error(f"Mailgun template email failed: {e}")
			raise EmailDeliveryError(f"Mailgun template email failed: {e}")
	
	async def validate_email(self, email: str) -> Dict[str, Any]:
		"""Validate email using Mailgun validation"""
		try:
			async with aiohttp.ClientSession() as session:
				auth = aiohttp.BasicAuth('api', self.api_key)
				async with session.get(
					f"https://api.mailgun.net/v4/address/validate",
					auth=auth,
					params={'address': email}
				) as response:
					result = await response.json()
					
					return {
						'is_valid': result.get('is_valid', False),
						'email': email,
						'provider': 'mailgun',
						'details': result
					}
		
		except Exception as e:
			self.logger.warning(f"Email validation failed: {e}")
			return {
				'is_valid': '@' in email and '.' in email,
				'email': email,
				'error': str(e)
			}


class SESEmailService(EmailService):
	"""Amazon SES email service implementation"""
	
	def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, 
				 region: str, from_email: str, from_name: str = None):
		if boto3 is None:
			raise EmailServiceError("boto3 is required to initialize Amazon SES email service")
		self.from_email = from_email
		self.from_name = from_name or "APG Billing"
		self.logger = logging.getLogger(f"{__name__}.SESEmailService")
		
		# Initialize SES client
		self.client = boto3.client(
			'ses',
			aws_access_key_id=aws_access_key_id,
			aws_secret_access_key=aws_secret_access_key,
			region_name=region
		)
	
	async def send_email(self, to_email: str, subject: str, html_content: str, 
						text_content: str = None, attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Send email via Amazon SES"""
		try:
			# Prepare email
			destination = {'ToAddresses': [to_email]}
			
			message = {
				'Subject': {'Data': subject, 'Charset': 'UTF-8'},
				'Body': {}
			}
			
			if text_content:
				message['Body']['Text'] = {'Data': text_content, 'Charset': 'UTF-8'}
			
			if html_content:
				message['Body']['Html'] = {'Data': html_content, 'Charset': 'UTF-8'}
			
			source = f"{self.from_name} <{self.from_email}>"
			
			# Send email
			response = self.client.send_email(
				Source=source,
				Destination=destination,
				Message=message
			)
			
			return {
				'success': True,
				'message_id': response['MessageId'],
				'provider': 'ses'
			}
		
		except Exception as e:
			self.logger.error(f"SES email failed: {e}")
			raise EmailDeliveryError(f"SES email failed: {e}")
	
	async def send_template_email(self, to_email: str, template_id: str, 
								 template_data: Dict[str, Any], attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Send templated email via SES"""
		try:
			response = self.client.send_templated_email(
				Source=f"{self.from_name} <{self.from_email}>",
				Destination={'ToAddresses': [to_email]},
				Template=template_id,
				TemplateData=json.dumps(template_data)
			)
			
			return {
				'success': True,
				'message_id': response['MessageId'],
				'provider': 'ses'
			}
		
		except Exception as e:
			self.logger.error(f"SES template email failed: {e}")
			raise EmailDeliveryError(f"SES template email failed: {e}")
	
	async def validate_email(self, email: str) -> Dict[str, Any]:
		"""Basic email validation for SES"""
		# SES doesn't have built-in validation, so we'll do basic checks
		is_valid = '@' in email and '.' in email.split('@')[1]
		
		return {
			'is_valid': is_valid,
			'email': email,
			'provider': 'ses'
		}


class BillingEmailManager:
	"""Manager for billing-specific email operations"""
	
	def __init__(self, email_service: EmailService):
		self.email_service = email_service
		self.logger = logging.getLogger(f"{__name__}.BillingEmailManager")
	
	async def send_invoice_email(self, customer: BLCustomer, invoice: BLInvoice, 
								pdf_content: bytes = None) -> Dict[str, Any]:
		"""Send invoice email to customer"""
		try:
			# Prepare email content
			subject = f"Invoice {invoice.invoice_number} from {invoice.tenant_id}"
			
			# HTML content
			html_content = f"""
			<html>
			<head>
				<style>
					body {{ font-family: Arial, sans-serif; margin: 20px; }}
					.header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
					.invoice-details {{ margin: 20px 0; }}
					.amount {{ font-size: 24px; font-weight: bold; color: #28a745; }}
					.footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
				</style>
			</head>
			<body>
				<div class="header">
					<h1>Invoice from APG Billing</h1>
					<p>Dear {customer.name},</p>
					<p>Your invoice is ready for review.</p>
				</div>
				
				<div class="invoice-details">
					<h2>Invoice Details</h2>
					<p><strong>Invoice Number:</strong> {invoice.invoice_number}</p>
					<p><strong>Invoice Date:</strong> {invoice.invoice_date.strftime('%B %d, %Y')}</p>
					<p><strong>Due Date:</strong> {invoice.due_date.strftime('%B %d, %Y')}</p>
					<p><strong>Amount Due:</strong> <span class="amount">{invoice.currency.value} {invoice.amount_due}</span></p>
				</div>
				
				<div class="payment-info">
					<h3>Payment Information</h3>
					<p>Please pay this invoice by the due date to avoid late fees.</p>
					<p>You can pay online by logging into your account or clicking the payment link below.</p>
				</div>
				
				<div class="footer">
					<p>Thank you for your business!</p>
					<p>If you have any questions about this invoice, please contact our billing team.</p>
				</div>
			</body>
			</html>
			"""
			
			# Text content fallback
			text_content = f"""
			Invoice from APG Billing
			
			Dear {customer.name},
			
			Your invoice is ready for review.
			
			Invoice Details:
			- Invoice Number: {invoice.invoice_number}
			- Invoice Date: {invoice.invoice_date.strftime('%B %d, %Y')}
			- Due Date: {invoice.due_date.strftime('%B %d, %Y')}
			- Amount Due: {invoice.currency.value} {invoice.amount_due}
			
			Please pay this invoice by the due date to avoid late fees.
			
			Thank you for your business!
			"""
			
			# Prepare attachments
			attachments = []
			if pdf_content:
				import base64
				attachments.append({
					'content': base64.b64encode(pdf_content).decode(),
					'filename': f'invoice_{invoice.invoice_number}.pdf',
					'content_type': 'application/pdf'
				})
			
			# Send email
			result = await self.email_service.send_email(
				to_email=customer.email,
				subject=subject,
				html_content=html_content,
				text_content=text_content,
				attachments=attachments
			)
			
			self.logger.info(f"Invoice email sent to {customer.email} for invoice {invoice.invoice_number}")
			return result
		
		except Exception as e:
			self.logger.error(f"Failed to send invoice email: {e}")
			raise EmailDeliveryError(f"Failed to send invoice email: {e}")
	
	async def send_payment_confirmation_email(self, customer: BLCustomer, payment: BLPayment, 
											 invoice: BLInvoice = None) -> Dict[str, Any]:
		"""Send payment confirmation email"""
		try:
			subject = f"Payment Confirmation - {payment.id}"
			
			html_content = f"""
			<html>
			<body style="font-family: Arial, sans-serif; margin: 20px;">
				<div style="background-color: #d4edda; padding: 20px; border-radius: 5px; border: 1px solid #c3e6cb;">
					<h1 style="color: #155724;">Payment Confirmed</h1>
					<p>Dear {customer.name},</p>
					<p>We have successfully received your payment.</p>
				</div>
				
				<div style="margin: 20px 0;">
					<h2>Payment Details</h2>
					<p><strong>Payment ID:</strong> {payment.id}</p>
					<p><strong>Amount:</strong> {payment.currency.value} {payment.amount}</p>
					<p><strong>Payment Date:</strong> {payment.processed_at.strftime('%B %d, %Y') if payment.processed_at else 'Processing'}</p>
					<p><strong>Payment Method:</strong> {payment.payment_method}</p>
					{f'<p><strong>Invoice:</strong> {invoice.invoice_number}</p>' if invoice else ''}
				</div>
				
				<p>Thank you for your payment!</p>
			</body>
			</html>
			"""
			
			result = await self.email_service.send_email(
				to_email=customer.email,
				subject=subject,
				html_content=html_content
			)
			
			self.logger.info(f"Payment confirmation sent to {customer.email} for payment {payment.id}")
			return result
		
		except Exception as e:
			self.logger.error(f"Failed to send payment confirmation: {e}")
			raise EmailDeliveryError(f"Failed to send payment confirmation: {e}")
	
	async def send_payment_failed_email(self, customer: BLCustomer, payment: BLPayment, 
									   failure_reason: str) -> Dict[str, Any]:
		"""Send payment failure notification"""
		try:
			subject = f"Payment Failed - Action Required"
			
			html_content = f"""
			<html>
			<body style="font-family: Arial, sans-serif; margin: 20px;">
				<div style="background-color: #f8d7da; padding: 20px; border-radius: 5px; border: 1px solid #f5c6cb;">
					<h1 style="color: #721c24;">Payment Failed</h1>
					<p>Dear {customer.name},</p>
					<p>We were unable to process your payment. Please review the details below and take action.</p>
				</div>
				
				<div style="margin: 20px 0;">
					<h2>Payment Details</h2>
					<p><strong>Payment ID:</strong> {payment.id}</p>
					<p><strong>Amount:</strong> {payment.currency.value} {payment.amount}</p>
					<p><strong>Failure Reason:</strong> {failure_reason}</p>
				</div>
				
				<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border: 1px solid #ffeaa7;">
					<h3>What to do next:</h3>
					<ul>
						<li>Check your payment method details</li>
						<li>Ensure sufficient funds are available</li>
						<li>Contact your bank if needed</li>
						<li>Try the payment again</li>
					</ul>
				</div>
				
				<p>If you continue to experience issues, please contact our support team.</p>
			</body>
			</html>
			"""
			
			result = await self.email_service.send_email(
				to_email=customer.email,
				subject=subject,
				html_content=html_content
			)
			
			self.logger.info(f"Payment failure notification sent to {customer.email}")
			return result
		
		except Exception as e:
			self.logger.error(f"Failed to send payment failure email: {e}")
			raise EmailDeliveryError(f"Failed to send payment failure email: {e}")


class EmailServiceManager:
	"""Manager for multiple email services"""
	
	def __init__(self):
		self.services: Dict[str, EmailService] = {}
		self.default_service = None
		self.logger = logging.getLogger(f"{__name__}.EmailServiceManager")
	
	def register_service(self, name: str, service: EmailService, is_default: bool = False):
		"""Register an email service"""
		self.services[name] = service
		if is_default or not self.default_service:
			self.default_service = name
		self.logger.info(f"Registered email service: {name}")
	
	def get_service(self, name: str = None) -> Optional[EmailService]:
		"""Get email service by name or default"""
		if name:
			return self.services.get(name)
		elif self.default_service:
			return self.services.get(self.default_service)
		return None
	
	def get_billing_email_manager(self, service_name: str = None) -> BillingEmailManager:
		"""Get billing email manager with specified service"""
		service = self.get_service(service_name)
		if not service:
			raise EmailServiceError("No email service available")
		return BillingEmailManager(service)


# Global email service manager
_email_manager_instance: Optional[EmailServiceManager] = None

def get_email_service_manager() -> EmailServiceManager:
	"""Get global email service manager instance"""
	global _email_manager_instance
	if _email_manager_instance is None:
		_email_manager_instance = EmailServiceManager()
		
		# Initialize with available services
		import os
		
		# SendGrid
		sendgrid_key = os.getenv('SENDGRID_API_KEY')
		sendgrid_from = os.getenv('SENDGRID_FROM_EMAIL')
		if sendgrid_key and sendgrid_from:
			sendgrid_service = SendGridEmailService(
				api_key=sendgrid_key,
				from_email=sendgrid_from,
				from_name=os.getenv('SENDGRID_FROM_NAME', 'APG Billing')
			)
			_email_manager_instance.register_service('sendgrid', sendgrid_service, is_default=True)
		
		# Mailgun
		mailgun_key = os.getenv('MAILGUN_API_KEY')
		mailgun_domain = os.getenv('MAILGUN_DOMAIN')
		mailgun_from = os.getenv('MAILGUN_FROM_EMAIL')
		if mailgun_key and mailgun_domain and mailgun_from:
			mailgun_service = MailgunEmailService(
				api_key=mailgun_key,
				domain=mailgun_domain,
				from_email=mailgun_from,
				from_name=os.getenv('MAILGUN_FROM_NAME', 'APG Billing')
			)
			_email_manager_instance.register_service('mailgun', mailgun_service)
		
		# Amazon SES
		aws_key = os.getenv('AWS_ACCESS_KEY_ID')
		aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
		aws_region = os.getenv('AWS_SES_REGION', 'us-east-1')
		ses_from = os.getenv('SES_FROM_EMAIL')
		if aws_key and aws_secret and ses_from:
			ses_service = SESEmailService(
				aws_access_key_id=aws_key,
				aws_secret_access_key=aws_secret,
				region=aws_region,
				from_email=ses_from,
				from_name=os.getenv('SES_FROM_NAME', 'APG Billing')
			)
			_email_manager_instance.register_service('ses', ses_service)
	
	return _email_manager_instance


__all__ = [
	'EmailService',
	'SendGridEmailService',
	'MailgunEmailService', 
	'SESEmailService',
	'BillingEmailManager',
	'EmailServiceManager',
	'get_email_service_manager',
	'EmailServiceError',
	'EmailDeliveryError'
]
