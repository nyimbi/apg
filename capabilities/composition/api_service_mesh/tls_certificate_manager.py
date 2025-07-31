"""
APG API Service Mesh - TLS/mTLS Certificate Management Engine

Complete TLS certificate lifecycle management, automatic rotation, 
and mutual TLS authentication for service mesh security.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import os
import ssl
import hashlib
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import httpx
import aiofiles
from cryptography import x509
from cryptography.x509.oid import NameOID, SignatureAlgorithmOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from uuid_extensions import uuid7str

from .models import SMCertificate, SMService, SMEndpoint


class CertificateType(str, Enum):
	"""Certificate types."""
	ROOT_CA = "root_ca"
	INTERMEDIATE_CA = "intermediate_ca"
	SERVICE_CERT = "service_cert"
	CLIENT_CERT = "client_cert"
	WILDCARD_CERT = "wildcard_cert"


class CertificateStatus(str, Enum):
	"""Certificate status."""
	ACTIVE = "active"
	EXPIRING = "expiring"
	EXPIRED = "expired"
	REVOKED = "revoked"
	PENDING = "pending"


@dataclass
class CertificateRequest:
	"""Certificate request data."""
	common_name: str
	sans: List[str]
	organization: str
	organizational_unit: str
	country: str
	state: str
	locality: str
	key_size: int = 2048
	validity_days: int = 365
	cert_type: CertificateType = CertificateType.SERVICE_CERT


@dataclass
class CertificateBundle:
	"""Complete certificate bundle."""
	certificate: str  # PEM encoded certificate
	private_key: str  # PEM encoded private key
	ca_certificate: str  # PEM encoded CA certificate
	certificate_chain: str  # Full certificate chain
	fingerprint: str
	serial_number: str
	not_before: datetime
	not_after: datetime


class TLSCertificateManager:
	"""Complete TLS/mTLS certificate management system."""
	
	def __init__(
		self,
		db_session: AsyncSession,
		cert_store_path: str = "/etc/service-mesh/certs",
		ca_cert_path: Optional[str] = None,
		ca_key_path: Optional[str] = None,
		auto_renewal_days: int = 30
	):
		self.db_session = db_session
		self.cert_store_path = Path(cert_store_path)
		self.ca_cert_path = ca_cert_path
		self.ca_key_path = ca_key_path
		self.auto_renewal_days = auto_renewal_days
		
		# Ensure certificate store exists
		self.cert_store_path.mkdir(parents=True, exist_ok=True)
		
		# Certificate cache
		self._cert_cache: Dict[str, CertificateBundle] = {}
		
		# CA certificate and key
		self._ca_cert: Optional[x509.Certificate] = None
		self._ca_private_key: Optional[rsa.RSAPrivateKey] = None
	
	async def initialize(self) -> None:
		"""Initialize certificate manager."""
		await self._load_ca_certificate()
		await self._start_certificate_monitor()
	
	async def _load_ca_certificate(self) -> None:
		"""Load CA certificate and private key."""
		try:
			if self.ca_cert_path and self.ca_key_path:
				# Load existing CA
				async with aiofiles.open(self.ca_cert_path, 'rb') as f:
					ca_cert_data = await f.read()
				self._ca_cert = x509.load_pem_x509_certificate(ca_cert_data)
				
				async with aiofiles.open(self.ca_key_path, 'rb') as f:
					ca_key_data = await f.read()
				self._ca_private_key = serialization.load_pem_private_key(
					ca_key_data, password=None
				)
			else:
				# Generate new CA
				await self._generate_root_ca()
		except Exception as e:
			print(f"Error loading CA certificate: {e}")
			await self._generate_root_ca()
	
	async def _generate_root_ca(self) -> Tuple[str, str]:
		"""Generate root CA certificate and private key."""
		# Generate private key
		private_key = rsa.generate_private_key(
			public_exponent=65537,
			key_size=4096
		)
		
		# Create certificate
		subject = issuer = x509.Name([
			x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
			x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
			x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
			x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Service Mesh CA"),
			x509.NameAttribute(NameOID.COMMON_NAME, "Service Mesh Root CA"),
		])
		
		cert = x509.CertificateBuilder().subject_name(
			subject
		).issuer_name(
			issuer
		).public_key(
			private_key.public_key()
		).serial_number(
			x509.random_serial_number()
		).not_valid_before(
			datetime.now(timezone.utc)
		).not_valid_after(
			datetime.now(timezone.utc) + timedelta(days=3650)  # 10 years
		).add_extension(
			x509.SubjectAlternativeName([
				x509.DNSName("service-mesh-ca"),
			]),
			critical=False,
		).add_extension(
			x509.BasicConstraints(ca=True, path_length=None),
			critical=True,
		).add_extension(
			x509.KeyUsage(
				key_cert_sign=True,
				crl_sign=True,
				digital_signature=False,
				key_encipherment=False,
				key_agreement=False,
				content_commitment=False,
				data_encipherment=False,
				encipher_only=False,
				decipher_only=False,
			),
			critical=True,
		).sign(private_key, hashes.SHA256())
		
		# Store CA certificate and key
		self._ca_cert = cert
		self._ca_private_key = private_key
		
		# Save to files
		ca_cert_path = self.cert_store_path / "ca-cert.pem"
		ca_key_path = self.cert_store_path / "ca-key.pem"
		
		async with aiofiles.open(ca_cert_path, 'wb') as f:
			await f.write(cert.public_bytes(Encoding.PEM))
		
		async with aiofiles.open(ca_key_path, 'wb') as f:
			await f.write(private_key.private_bytes(
				encoding=Encoding.PEM,
				format=PrivateFormat.PKCS8,
				encryption_algorithm=NoEncryption()
			))
		
		# Store in database
		cert_pem = cert.public_bytes(Encoding.PEM).decode('utf-8')
		key_pem = private_key.private_bytes(
			encoding=Encoding.PEM,
			format=PrivateFormat.PKCS8,
			encryption_algorithm=NoEncryption()
		).decode('utf-8')
		
		await self._store_certificate_in_db(
			certificate_id=uuid7str(),
			cert_type=CertificateType.ROOT_CA,
			common_name="Service Mesh Root CA",
			certificate_pem=cert_pem,
			private_key_pem=key_pem,
			not_before=cert.not_valid_before,
			not_after=cert.not_valid_after,
			sans=["service-mesh-ca"]
		)
		
		return cert_pem, key_pem
	
	async def generate_service_certificate(
		self,
		service_name: str,
		namespace: str = "default",
		sans: Optional[List[str]] = None,
		validity_days: int = 90
	) -> CertificateBundle:
		"""Generate certificate for a service."""
		if not self._ca_cert or not self._ca_private_key:
			raise RuntimeError("CA certificate not loaded")
		
		# Build common name and SANs
		common_name = f"{service_name}.{namespace}.svc.cluster.local"
		subject_alt_names = sans or []
		subject_alt_names.extend([
			common_name,
			f"{service_name}.{namespace}",
			f"{service_name}.{namespace}.svc",
			service_name
		])
		
		# Remove duplicates
		subject_alt_names = list(set(subject_alt_names))
		
		# Generate private key
		private_key = rsa.generate_private_key(
			public_exponent=65537,
			key_size=2048
		)
		
		# Create certificate
		subject = x509.Name([
			x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
			x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
			x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
			x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Service Mesh"),
			x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, namespace),
			x509.NameAttribute(NameOID.COMMON_NAME, common_name),
		])
		
		cert = x509.CertificateBuilder().subject_name(
			subject
		).issuer_name(
			self._ca_cert.subject
		).public_key(
			private_key.public_key()
		).serial_number(
			x509.random_serial_number()
		).not_valid_before(
			datetime.now(timezone.utc)
		).not_valid_after(
			datetime.now(timezone.utc) + timedelta(days=validity_days)
		).add_extension(
			x509.SubjectAlternativeName([
				x509.DNSName(san) for san in subject_alt_names
			]),
			critical=False,
		).add_extension(
			x509.BasicConstraints(ca=False, path_length=None),
			critical=True,
		).add_extension(
			x509.KeyUsage(
				key_cert_sign=False,
				crl_sign=False,
				digital_signature=True,
				key_encipherment=True,
				key_agreement=False,
				content_commitment=True,
				data_encipherment=False,
				encipher_only=False,
				decipher_only=False,
			),
			critical=True,
		).add_extension(
			x509.ExtendedKeyUsage([
				x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
				x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
			]),
			critical=True,
		).sign(self._ca_private_key, hashes.SHA256())
		
		# Create certificate bundle
		cert_pem = cert.public_bytes(Encoding.PEM).decode('utf-8')
		key_pem = private_key.private_bytes(
			encoding=Encoding.PEM,
			format=PrivateFormat.PKCS8,
			encryption_algorithm=NoEncryption()
		).decode('utf-8')
		ca_cert_pem = self._ca_cert.public_bytes(Encoding.PEM).decode('utf-8')
		
		fingerprint = hashlib.sha256(cert.public_bytes(Encoding.DER)).hexdigest()
		
		bundle = CertificateBundle(
			certificate=cert_pem,
			private_key=key_pem,
			ca_certificate=ca_cert_pem,
			certificate_chain=cert_pem + ca_cert_pem,
			fingerprint=fingerprint,
			serial_number=str(cert.serial_number),
			not_before=cert.not_valid_before,
			not_after=cert.not_valid_after
		)
		
		# Store in database
		certificate_id = uuid7str()
		await self._store_certificate_in_db(
			certificate_id=certificate_id,
			cert_type=CertificateType.SERVICE_CERT,
			common_name=common_name,
			certificate_pem=cert_pem,
			private_key_pem=key_pem,
			not_before=cert.not_valid_before,
			not_after=cert.not_valid_after,
			sans=subject_alt_names,
			fingerprint=fingerprint
		)
		
		# Cache the bundle
		self._cert_cache[service_name] = bundle
		
		# Save to files
		cert_dir = self.cert_store_path / namespace / service_name
		cert_dir.mkdir(parents=True, exist_ok=True)
		
		async with aiofiles.open(cert_dir / "tls.crt", 'w') as f:
			await f.write(cert_pem)
		
		async with aiofiles.open(cert_dir / "tls.key", 'w') as f:
			await f.write(key_pem)
		
		async with aiofiles.open(cert_dir / "ca.crt", 'w') as f:
			await f.write(ca_cert_pem)
		
		return bundle
	
	async def get_certificate_bundle(self, service_name: str) -> Optional[CertificateBundle]:
		"""Get certificate bundle for a service."""
		# Check cache first
		if service_name in self._cert_cache:
			bundle = self._cert_cache[service_name]
			# Check if certificate is still valid
			if bundle.not_after > datetime.now(timezone.utc):
				return bundle
		
		# Load from database
		result = await self.db_session.execute(
			select(SMCertificate).where(
				and_(
					SMCertificate.common_name.contains(service_name),
					SMCertificate.status == "active",
					SMCertificate.not_after > datetime.now(timezone.utc)
				)
			)
		)
		cert_record = result.scalars().first()
		
		if not cert_record:
			return None
		
		# Create bundle from database record
		bundle = CertificateBundle(
			certificate=cert_record.certificate_pem,
			private_key=cert_record.private_key_pem or "",
			ca_certificate=self._ca_cert.public_bytes(Encoding.PEM).decode('utf-8') if self._ca_cert else "",
			certificate_chain=cert_record.certificate_pem + (self._ca_cert.public_bytes(Encoding.PEM).decode('utf-8') if self._ca_cert else ""),
			fingerprint=cert_record.fingerprint or "",
			serial_number=cert_record.serial_number or "",
			not_before=cert_record.not_before,
			not_after=cert_record.not_after
		)
		
		# Update cache
		self._cert_cache[service_name] = bundle
		
		return bundle
	
	async def revoke_certificate(self, certificate_id: str, reason: str = "unspecified") -> bool:
		"""Revoke a certificate."""
		try:
			await self.db_session.execute(
				update(SMCertificate).where(
					SMCertificate.id == certificate_id
				).values(
					status="revoked",
					revocation_reason=reason,
					revoked_at=datetime.now(timezone.utc)
				)
			)
			await self.db_session.commit()
			
			# Remove from cache
			for service_name, bundle in list(self._cert_cache.items()):
				if bundle.serial_number in certificate_id:
					del self._cert_cache[service_name]
			
			return True
		except Exception as e:
			print(f"Error revoking certificate: {e}")
			await self.db_session.rollback()
			return False
	
	async def renew_certificate(
		self,
		service_name: str,
		namespace: str = "default",
		validity_days: int = 90
	) -> Optional[CertificateBundle]:
		"""Renew certificate for a service."""
		# Revoke old certificate
		old_bundle = await self.get_certificate_bundle(service_name)
		if old_bundle:
			result = await self.db_session.execute(
				select(SMCertificate).where(
					SMCertificate.fingerprint == old_bundle.fingerprint
				)
			)
			old_cert = result.scalars().first()
			if old_cert:
				await self.revoke_certificate(old_cert.id, "superseded")
		
		# Generate new certificate
		return await self.generate_service_certificate(
			service_name=service_name,
			namespace=namespace,
			validity_days=validity_days
		)
	
	async def check_certificate_expiration(self) -> List[Dict[str, Any]]:
		"""Check for certificates expiring soon."""
		expiration_threshold = datetime.now(timezone.utc) + timedelta(days=self.auto_renewal_days)
		
		result = await self.db_session.execute(
			select(SMCertificate).where(
				and_(
					SMCertificate.status == "active",
					SMCertificate.not_after <= expiration_threshold
				)
			)
		)
		
		expiring_certs = []
		for cert in result.scalars():
			days_until_expiry = (cert.not_after - datetime.now(timezone.utc)).days
			expiring_certs.append({
				"certificate_id": cert.id,
				"common_name": cert.common_name,
				"not_after": cert.not_after,
				"days_until_expiry": days_until_expiry,
				"fingerprint": cert.fingerprint
			})
		
		return expiring_certs
	
	async def auto_renew_certificates(self) -> Dict[str, Any]:
		"""Automatically renew expiring certificates."""
		expiring_certs = await self.check_certificate_expiration()
		renewal_results = {
			"renewed": [],
			"failed": [],
			"total_checked": len(expiring_certs)
		}
		
		for cert_info in expiring_certs:
			try:
				# Extract service name from common name
				common_name = cert_info["common_name"]
				if ".svc.cluster.local" in common_name:
					service_name = common_name.split(".")[0]
					namespace = common_name.split(".")[1]
				else:
					service_name = common_name
					namespace = "default"
				
				# Renew certificate
				new_bundle = await self.renew_certificate(service_name, namespace)
				if new_bundle:
					renewal_results["renewed"].append({
						"service_name": service_name,
						"namespace": namespace,
						"old_fingerprint": cert_info["fingerprint"],
						"new_fingerprint": new_bundle.fingerprint,
						"new_expiry": new_bundle.not_after
					})
				else:
					renewal_results["failed"].append({
						"service_name": service_name,
						"error": "Failed to generate new certificate"
					})
			except Exception as e:
				renewal_results["failed"].append({
					"common_name": cert_info["common_name"],
					"error": str(e)
				})
		
		return renewal_results
	
	async def validate_certificate_chain(self, certificate_pem: str) -> Dict[str, Any]:
		"""Validate certificate chain."""
		try:
			cert = x509.load_pem_x509_certificate(certificate_pem.encode())
			
			# Basic validation
			now = datetime.now(timezone.utc)
			is_valid = cert.not_valid_before <= now <= cert.not_valid_after
			
			# Check if signed by our CA
			is_ca_signed = False
			if self._ca_cert:
				try:
					self._ca_cert.public_key().verify(
						cert.signature,
						cert.tbs_certificate_bytes,
						padding.PKCS1v15(),
						cert.signature_hash_algorithm
					)
					is_ca_signed = True
				except Exception:
					pass
			
			return {
				"is_valid": is_valid,
				"is_ca_signed": is_ca_signed,
				"not_before": cert.not_valid_before,
				"not_after": cert.not_valid_after,
				"serial_number": str(cert.serial_number),
				"fingerprint": hashlib.sha256(cert.public_bytes(Encoding.DER)).hexdigest(),
				"subject": cert.subject.rfc4514_string(),
				"issuer": cert.issuer.rfc4514_string(),
				"sans": [san.value for san in cert.extensions.get_extension_for_oid(
					x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
				).value] if cert.extensions else []
			}
		except Exception as e:
			return {
				"is_valid": False,
				"error": str(e)
			}
	
	async def create_tls_context(
		self,
		service_name: str,
		verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED
	) -> ssl.SSLContext:
		"""Create SSL context for TLS/mTLS."""
		bundle = await self.get_certificate_bundle(service_name)
		if not bundle:
			raise ValueError(f"No certificate found for service: {service_name}")
		
		# Create SSL context
		context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
		context.check_hostname = False
		context.verify_mode = verify_mode
		
		# Load certificate and key
		context.load_cert_chain(
			certfile=None,
			keyfile=None,
			password=None
		)
		
		# For mTLS, load CA certificate
		if verify_mode == ssl.CERT_REQUIRED:
			context.load_verify_locations(cadata=bundle.ca_certificate)
		
		return context
	
	async def _store_certificate_in_db(
		self,
		certificate_id: str,
		cert_type: CertificateType,
		common_name: str,
		certificate_pem: str,
		private_key_pem: str,
		not_before: datetime,
		not_after: datetime,
		sans: Optional[List[str]] = None,
		fingerprint: Optional[str] = None
	) -> None:
		"""Store certificate in database."""
		cert_record = SMCertificate(
			id=certificate_id,
			certificate_type=cert_type.value,
			common_name=common_name,
			certificate_pem=certificate_pem,
			private_key_pem=private_key_pem,
			not_before=not_before,
			not_after=not_after,
			subject_alternative_names=sans or [],
			fingerprint=fingerprint,
			serial_number=None,
			status="active",
			created_at=datetime.now(timezone.utc)
		)
		
		self.db_session.add(cert_record)
		await self.db_session.commit()
	
	async def _start_certificate_monitor(self) -> None:
		"""Start certificate monitoring background task."""
		asyncio.create_task(self._certificate_monitor_loop())
	
	async def _certificate_monitor_loop(self) -> None:
		"""Background task to monitor and auto-renew certificates."""
		while True:
			try:
				await asyncio.sleep(3600)  # Check every hour
				await self.auto_renew_certificates()
			except Exception as e:
				print(f"Certificate monitor error: {e}")
				await asyncio.sleep(300)  # Wait 5 minutes on error


# =============================================================================
# mTLS Policy Enforcement
# =============================================================================

class MTLSPolicyEngine:
	"""Mutual TLS policy enforcement engine."""
	
	def __init__(self, cert_manager: TLSCertificateManager):
		self.cert_manager = cert_manager
		self.mtls_policies: Dict[str, Dict[str, Any]] = {}
	
	async def create_mtls_policy(
		self,
		service_name: str,
		namespace: str,
		required_services: List[str],
		allowed_clients: Optional[List[str]] = None,
		strict_mode: bool = True
	) -> str:
		"""Create mTLS policy for a service."""
		policy_id = uuid7str()
		
		policy = {
			"id": policy_id,
			"service_name": service_name,
			"namespace": namespace,
			"required_services": required_services,
			"allowed_clients": allowed_clients or [],
			"strict_mode": strict_mode,
			"created_at": datetime.now(timezone.utc)
		}
		
		self.mtls_policies[f"{namespace}/{service_name}"] = policy
		return policy_id
	
	async def enforce_mtls_policy(
		self,
		service_name: str,
		namespace: str,
		client_cert: str
	) -> Dict[str, Any]:
		"""Enforce mTLS policy for incoming request."""
		policy_key = f"{namespace}/{service_name}"
		policy = self.mtls_policies.get(policy_key)
		
		if not policy:
			return {"allowed": True, "reason": "No mTLS policy defined"}
		
		# Validate client certificate
		cert_validation = await self.cert_manager.validate_certificate_chain(client_cert)
		if not cert_validation["is_valid"]:
			return {
				"allowed": False,
				"reason": "Invalid client certificate",
				"details": cert_validation
			}
		
		# Check if client is in allowed list
		if policy["allowed_clients"]:
			client_cn = cert_validation.get("subject", "")
			allowed = any(allowed_client in client_cn for allowed_client in policy["allowed_clients"])
			if not allowed:
				return {
					"allowed": False,
					"reason": "Client not in allowed list",
					"client_cn": client_cn
				}
		
		return {
			"allowed": True,
			"reason": "mTLS policy satisfied",
			"client_cn": cert_validation.get("subject", ""),
			"policy_id": policy["id"]
		}