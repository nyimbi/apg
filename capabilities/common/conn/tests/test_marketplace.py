"""
Tests for APG Connection Management Marketplace functionality
Comprehensive testing of marketplace integration, capability management, and installation

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from ..marketplace import (
	MarketplaceClient, CapabilityInstaller, MarketplaceManager,
	MarketplaceSearchQuery, MarketplaceCapability, InstalledCapability,
	CapabilityType, CapabilityStatus, LicenseType, InstallationStatus,
	CapabilityAuthor, CapabilityRating, CapabilityStats, CapabilityVersion,
	search_marketplace_capabilities, install_marketplace_capability,
	get_marketplace_recommendations
)


@pytest.fixture
def marketplace_client():
	"""Marketplace client instance"""
	return MarketplaceClient(marketplace_url="https://test.marketplace.com", api_key="test_key")


@pytest.fixture
def temp_install_dir():
	"""Temporary installation directory"""
	temp_dir = tempfile.mkdtemp()
	yield temp_dir
	shutil.rmtree(temp_dir)


@pytest.fixture
def capability_installer(temp_install_dir):
	"""Capability installer instance"""
	return CapabilityInstaller(installation_dir=temp_install_dir)


@pytest.fixture
def marketplace_manager():
	"""Marketplace manager instance"""
	return MarketplaceManager(marketplace_url="https://test.marketplace.com", api_key="test_key")


@pytest.fixture
def sample_capability():
	"""Sample marketplace capability"""
	return MarketplaceCapability(
		id="test-capability",
		name="Test Capability",
		description="A test capability for unit testing",
		capability_type=CapabilityType.CONNECTOR,
		status=CapabilityStatus.PUBLISHED,
		author=CapabilityAuthor(name="Test Author", email="test@example.com", verified=True),
		license=LicenseType.OPEN_SOURCE,
		current_version="1.0.0",
		tags=["test", "connector"],
		categories=["Testing"],
		rating=CapabilityRating(average_rating=4.5, total_reviews=10),
		stats=CapabilityStats(downloads=100, installations=50)
	)


@pytest.fixture
def sample_search_query():
	"""Sample search query"""
	return MarketplaceSearchQuery(
		query="database",
		capability_type=CapabilityType.CONNECTOR,
		tags=["sql", "database"],
		min_rating=4.0,
		limit=10
	)


class TestMarketplaceClient:
	"""Test marketplace client functionality"""

	def test_client_initialization(self, marketplace_client):
		"""Test client initialization"""
		assert marketplace_client.marketplace_url == "https://test.marketplace.com"
		assert marketplace_client.api_key == "test_key"
		assert marketplace_client.timeout == 30
		assert marketplace_client._http_client is None

	@pytest.mark.asyncio
	async def test_http_client_creation(self, marketplace_client):
		"""Test HTTP client creation"""
		with patch('httpx.AsyncClient') as mock_client:
			client = await marketplace_client._get_http_client()
			assert client is not None
			mock_client.assert_called_once()

	@pytest.mark.asyncio
	async def test_search_capabilities_local_catalog(self, marketplace_client, sample_search_query):
		"""Test capability search with bundled local catalog data"""
		results = await marketplace_client.search_capabilities(sample_search_query)

		assert "capabilities" in results
		assert "total" in results
		assert isinstance(results["capabilities"], list)
		assert results["total"] >= 0

	@pytest.mark.asyncio
	async def test_get_capability_local_catalog(self, marketplace_client):
		"""Test getting capability details from bundled local catalog data"""
		capability = await marketplace_client.get_capability("test-capability")

		assert isinstance(capability, MarketplaceCapability)
		assert capability.id == "test-capability"
		assert capability.name == "Test Capability"
		assert capability.capability_type == CapabilityType.CONNECTOR

	@pytest.mark.asyncio
	async def test_get_capability_versions_local_catalog(self, marketplace_client):
		"""Test getting capability versions from bundled local catalog data"""
		versions = await marketplace_client.get_capability_versions("test-capability")

		assert isinstance(versions, list)
		assert len(versions) >= 1
		assert all(isinstance(v, CapabilityVersion) for v in versions)

	@pytest.mark.asyncio
	async def test_client_close(self, marketplace_client):
		"""Test client cleanup"""
		# Create a mock client
		mock_client = AsyncMock()
		marketplace_client._http_client = mock_client

		await marketplace_client.close()

		mock_client.aclose.assert_called_once()
		assert marketplace_client._http_client is None

	def test_parse_capability(self, marketplace_client):
		"""Test parsing capability data"""
		data = {
			"id": "test-cap",
			"name": "Test Capability",
			"description": "Test description",
			"capability_type": "connector",
			"status": "published",
			"license": "open_source",
			"current_version": "1.0.0",
			"author": {
				"name": "Test Author",
				"email": "test@example.com",
				"verified": True
			},
			"rating": {
				"average_rating": 4.5,
				"total_reviews": 10
			},
			"stats": {
				"downloads": 100,
				"installations": 50
			}
		}

		capability = marketplace_client._parse_capability(data)

		assert capability.id == "test-cap"
		assert capability.name == "Test Capability"
		assert capability.capability_type == CapabilityType.CONNECTOR
		assert capability.author.verified == True
		assert capability.rating.average_rating == 4.5

	def test_parse_version(self, marketplace_client):
		"""Test parsing version data"""
		data = {
			"version": "1.0.0",
			"release_notes": "Initial release",
			"compatibility": {"apg": ">=1.0.0"},
			"dependencies": ["requests>=2.25.0"]
		}

		version = marketplace_client._parse_version(data)

		assert version.version == "1.0.0"
		assert version.release_notes == "Initial release"
		assert version.compatibility == {"apg": ">=1.0.0"}
		assert version.dependencies == ["requests>=2.25.0"]


class TestCapabilityInstaller:
	"""Test capability installation functionality"""

	def test_installer_initialization(self, capability_installer, temp_install_dir):
		"""Test installer initialization"""
		assert capability_installer.installation_dir == Path(temp_install_dir)
		assert capability_installer.installation_dir.exists()
		assert isinstance(capability_installer.installed_capabilities, dict)

	def test_manifest_file_creation(self, capability_installer):
		"""Test manifest file handling"""
		# Add a test capability
		test_capability = InstalledCapability(
			capability_id="test-cap",
			name="Test Capability",
			version="1.0.0",
			installation_path="/test/path",
			status=InstallationStatus.INSTALLED,
			installed_at=datetime.now(timezone.utc)
		)

		capability_installer.installed_capabilities["test-cap"] = test_capability
		capability_installer._save_installed_capabilities()

		# Check manifest file exists
		manifest_file = capability_installer.installation_dir / "manifest.json"
		assert manifest_file.exists()

		# Load and verify
		with open(manifest_file, 'r') as f:
			data = json.load(f)

		assert len(data["installed"]) == 1
		assert data["installed"][0]["capability_id"] == "test-cap"

	@pytest.mark.asyncio
	async def test_install_capability_mock(self, capability_installer):
		"""Test capability installation with mock client"""
		mock_client = Mock()
		mock_client.get_capability = AsyncMock(return_value=Mock(
			name="Test Capability",
			current_version="1.0.0"
		))
		mock_client.download_capability = AsyncMock(return_value=b"mock package data")
		mock_client.close = AsyncMock()

		installed = await capability_installer.install_capability(
			"test-cap", "1.0.0", mock_client
		)

		assert isinstance(installed, InstalledCapability)
		assert installed.capability_id == "test-cap"
		assert installed.status == InstallationStatus.INSTALLED
		assert "test-cap" in capability_installer.installed_capabilities

	@pytest.mark.asyncio
	async def test_install_already_installed(self, capability_installer):
		"""Test installing capability that's already installed"""
		# Pre-install a capability
		existing = InstalledCapability(
			capability_id="existing-cap",
			name="Existing Capability",
			version="1.0.0",
			installation_path="/test/path",
			status=InstallationStatus.INSTALLED,
			installed_at=datetime.now(timezone.utc)
		)
		capability_installer.installed_capabilities["existing-cap"] = existing

		mock_client = Mock()
		mock_client.close = AsyncMock()

		result = await capability_installer.install_capability(
			"existing-cap", "1.0.0", mock_client
		)

		assert result == existing

	@pytest.mark.asyncio
	async def test_uninstall_capability(self, capability_installer):
		"""Test capability uninstallation"""
		# Install a capability first
		test_capability = InstalledCapability(
			capability_id="uninstall-test",
			name="Uninstall Test",
			version="1.0.0",
			installation_path=str(capability_installer.installation_dir / "test"),
			status=InstallationStatus.INSTALLED,
			installed_at=datetime.now(timezone.utc)
		)

		# Create installation directory
		install_path = Path(test_capability.installation_path)
		install_path.mkdir(parents=True, exist_ok=True)

		capability_installer.installed_capabilities["uninstall-test"] = test_capability

		# Uninstall
		success = await capability_installer.uninstall_capability("uninstall-test")

		assert success == True
		assert "uninstall-test" not in capability_installer.installed_capabilities
		assert not install_path.exists()

	@pytest.mark.asyncio
	async def test_uninstall_non_existent(self, capability_installer):
		"""Test uninstalling non-existent capability"""
		success = await capability_installer.uninstall_capability("non-existent")
		assert success == False

	def test_get_installed_capabilities(self, capability_installer):
		"""Test getting list of installed capabilities"""
		# Add test capabilities
		cap1 = InstalledCapability(
			capability_id="cap1",
			name="Capability 1",
			version="1.0.0",
			installation_path="/test/path1",
			status=InstallationStatus.INSTALLED,
			installed_at=datetime.now(timezone.utc)
		)
		cap2 = InstalledCapability(
			capability_id="cap2",
			name="Capability 2",
			version="2.0.0",
			installation_path="/test/path2",
			status=InstallationStatus.INSTALLED,
			installed_at=datetime.now(timezone.utc)
		)

		capability_installer.installed_capabilities["cap1"] = cap1
		capability_installer.installed_capabilities["cap2"] = cap2

		installed = capability_installer.get_installed_capabilities()

		assert len(installed) == 2
		assert cap1 in installed
		assert cap2 in installed

	def test_get_capability_info(self, capability_installer):
		"""Test getting specific capability info"""
		test_cap = InstalledCapability(
			capability_id="info-test",
			name="Info Test",
			version="1.0.0",
			installation_path="/test/path",
			status=InstallationStatus.INSTALLED,
			installed_at=datetime.now(timezone.utc)
		)

		capability_installer.installed_capabilities["info-test"] = test_cap

		result = capability_installer.get_capability_info("info-test")
		assert result == test_cap

		result = capability_installer.get_capability_info("non-existent")
		assert result is None

	def test_is_capability_installed(self, capability_installer):
		"""Test checking if capability is installed"""
		test_cap = InstalledCapability(
			capability_id="installed-test",
			name="Installed Test",
			version="1.0.0",
			installation_path="/test/path",
			status=InstallationStatus.INSTALLED,
			installed_at=datetime.now(timezone.utc)
		)

		capability_installer.installed_capabilities["installed-test"] = test_cap

		assert capability_installer.is_capability_installed("installed-test") == True
		assert capability_installer.is_capability_installed("not-installed") == False

	def test_version_comparison(self, capability_installer):
		"""Test version comparison logic"""
		# Test basic version comparison (fallback)
		assert capability_installer._is_newer_version("2.0.0", "1.0.0") == True
		assert capability_installer._is_newer_version("1.0.0", "2.0.0") == False
		assert capability_installer._is_newer_version("1.0.0", "1.0.0") == False


class TestMarketplaceManager:
	"""Test marketplace manager functionality"""

	def test_manager_initialization(self, marketplace_manager):
		"""Test manager initialization"""
		assert isinstance(marketplace_manager.client, MarketplaceClient)
		assert isinstance(marketplace_manager.installer, CapabilityInstaller)

	@pytest.mark.asyncio
	async def test_search_capabilities_cached(self, marketplace_manager):
		"""Test cached capability search"""
		results = await marketplace_manager.search_capabilities(
			query="database",
			capability_type=CapabilityType.CONNECTOR
		)

		assert isinstance(results, dict)
		assert "capabilities" in results

	@pytest.mark.asyncio
	async def test_get_capability_details_cached(self, marketplace_manager):
		"""Test cached capability details"""
		capability = await marketplace_manager.get_capability_details("test-capability")

		assert isinstance(capability, MarketplaceCapability)
		assert capability.id == "test-capability"

	@pytest.mark.asyncio
	async def test_get_featured_capabilities(self, marketplace_manager):
		"""Test getting featured capabilities"""
		featured = await marketplace_manager.get_featured_capabilities()

		assert isinstance(featured, list)
		assert all(isinstance(cap, MarketplaceCapability) for cap in featured)

	@pytest.mark.asyncio
	async def test_get_recommendations(self, marketplace_manager):
		"""Test getting personalized recommendations"""
		recommendations = await marketplace_manager.get_recommendations("test-tenant")

		assert isinstance(recommendations, list)
		assert all(isinstance(cap, MarketplaceCapability) for cap in recommendations)

	def test_get_installed_capabilities(self, marketplace_manager):
		"""Test getting installed capabilities"""
		installed = marketplace_manager.get_installed_capabilities()

		assert isinstance(installed, list)
		assert all(isinstance(cap, InstalledCapability) for cap in installed)

	@pytest.mark.asyncio
	async def test_manager_close(self, marketplace_manager):
		"""Test manager cleanup"""
		with patch.object(marketplace_manager.client, 'close', new=AsyncMock()) as mock_close:
			await marketplace_manager.close()
			mock_close.assert_called_once()


class TestDataClasses:
	"""Test data classes and enums"""

	def test_capability_type_enum(self):
		"""Test CapabilityType enum"""
		assert CapabilityType.CONNECTOR.value == "connector"
		assert CapabilityType.TRANSFORMER.value == "transformer"
		assert len(list(CapabilityType)) >= 7

	def test_capability_status_enum(self):
		"""Test CapabilityStatus enum"""
		assert CapabilityStatus.PUBLISHED.value == "published"
		assert CapabilityStatus.DRAFT.value == "draft"
		assert len(list(CapabilityStatus)) >= 5

	def test_license_type_enum(self):
		"""Test LicenseType enum"""
		assert LicenseType.OPEN_SOURCE.value == "open_source"
		assert LicenseType.COMMERCIAL.value == "commercial"
		assert len(list(LicenseType)) >= 5

	def test_installation_status_enum(self):
		"""Test InstallationStatus enum"""
		assert InstallationStatus.INSTALLED.value == "installed"
		assert InstallationStatus.INSTALLING.value == "installing"
		assert len(list(InstallationStatus)) >= 5

	def test_capability_author_creation(self):
		"""Test CapabilityAuthor data class"""
		author = CapabilityAuthor(
			name="Test Author",
			email="test@example.com",
			organization="Test Org",
			verified=True
		)

		assert author.name == "Test Author"
		assert author.email == "test@example.com"
		assert author.organization == "Test Org"
		assert author.verified == True

	def test_capability_rating_creation(self):
		"""Test CapabilityRating data class"""
		rating = CapabilityRating(
			average_rating=4.5,
			total_reviews=100,
			five_star=60,
			four_star=25,
			three_star=10,
			two_star=3,
			one_star=2
		)

		assert rating.average_rating == 4.5
		assert rating.total_reviews == 100
		assert rating.five_star == 60

	def test_capability_version_creation(self):
		"""Test CapabilityVersion data class"""
		version = CapabilityVersion(
			version="1.0.0",
			release_notes="Initial release",
			compatibility={"apg": ">=1.0.0"},
			dependencies=["requests>=2.25.0"],
			breaking_changes=["Changed API format"],
			security_fixes=["Fixed auth bypass"]
		)

		assert version.version == "1.0.0"
		assert version.release_notes == "Initial release"
		assert "requests>=2.25.0" in version.dependencies
		assert len(version.breaking_changes) == 1

	def test_marketplace_search_query(self):
		"""Test MarketplaceSearchQuery data class"""
		query = MarketplaceSearchQuery(
			query="database connector",
			capability_type=CapabilityType.CONNECTOR,
			tags=["sql", "postgres"],
			min_rating=4.0,
			free_only=True,
			verified_only=True,
			sort_by="rating",
			limit=20
		)

		assert query.query == "database connector"
		assert query.capability_type == CapabilityType.CONNECTOR
		assert "sql" in query.tags
		assert query.min_rating == 4.0
		assert query.free_only == True

	def test_installed_capability_defaults(self):
		"""Test InstalledCapability default values"""
		capability = InstalledCapability(
			capability_id="test",
			name="Test",
			version="1.0.0",
			installation_path="/test",
			status=InstallationStatus.INSTALLED,
			installed_at=datetime.now(timezone.utc)
		)

		assert capability.auto_update == True
		assert capability.usage_count == 0
		assert capability.last_used is None
		assert isinstance(capability.config, dict)


class TestConvenienceFunctions:
	"""Test convenience functions"""

	@pytest.mark.asyncio
	async def test_search_marketplace_capabilities(self):
		"""Test convenience search function"""
		results = await search_marketplace_capabilities(
			query="database",
			capability_type="connector",
			tags=["sql"]
		)

		assert isinstance(results, dict)
		assert "capabilities" in results

	@pytest.mark.asyncio
	async def test_search_marketplace_capabilities_no_params(self):
		"""Test search without parameters"""
		results = await search_marketplace_capabilities()

		assert isinstance(results, dict)
		assert "capabilities" in results

	@pytest.mark.asyncio
	async def test_get_marketplace_recommendations(self):
		"""Test recommendations function"""
		recommendations = await get_marketplace_recommendations("test-tenant")

		assert isinstance(recommendations, list)
		assert all(isinstance(cap, MarketplaceCapability) for cap in recommendations)


class TestErrorHandling:
	"""Test error handling and edge cases"""

	@pytest.mark.asyncio
	async def test_install_with_client_error(self, capability_installer):
		"""Test installation with client error"""
		mock_client = Mock()
		mock_client.get_capability = AsyncMock(side_effect=Exception("API Error"))
		mock_client.close = AsyncMock()

		with pytest.raises(Exception):
			await capability_installer.install_capability("error-cap", "1.0.0", mock_client)

		# Should mark as failed
		if "error-cap" in capability_installer.installed_capabilities:
			assert capability_installer.installed_capabilities["error-cap"].status == InstallationStatus.FAILED

	@pytest.mark.asyncio
	async def test_search_with_invalid_params(self, marketplace_client):
		"""Test search with invalid parameters"""
		invalid_query = MarketplaceSearchQuery(
			capability_type="invalid_type",  # This will cause validation error
			min_rating=-1  # Invalid rating
		)

		# Should handle gracefully and return local catalog results
		results = await marketplace_client.search_capabilities(invalid_query)
		assert isinstance(results, dict)

	def test_capability_installer_invalid_path(self):
		"""Test installer with invalid installation path"""
		# This should work - installer creates directories
		installer = CapabilityInstaller("/non/existent/path/test")
		assert installer.installation_dir.exists()


class TestIntegration:
	"""Integration tests"""

	@pytest.mark.asyncio
	async def test_end_to_end_capability_lifecycle(self, temp_install_dir):
		"""Test complete capability lifecycle"""
		# Create installer and client
		installer = CapabilityInstaller(temp_install_dir)
		client = MarketplaceClient("https://test.marketplace.com")

		try:
			# Search for capabilities
			search_query = MarketplaceSearchQuery(query="test", limit=5)
			search_results = await client.search_capabilities(search_query)
			assert "capabilities" in search_results

			# Get capability details
			capability = await client.get_capability("test-capability")
			assert isinstance(capability, MarketplaceCapability)

			# Install capability from local catalog
			with patch.object(client, 'download_capability', new=AsyncMock(return_value=b"test data")):
				installed = await installer.install_capability("test-capability", "1.0.0", client)
				assert isinstance(installed, InstalledCapability)
				assert installed.status == InstallationStatus.INSTALLED

			# Check installed
			assert installer.is_capability_installed("test-capability")

			# Get installed list
			installed_list = installer.get_installed_capabilities()
			assert len(installed_list) == 1

			# Uninstall
			success = await installer.uninstall_capability("test-capability")
			assert success == True
			assert not installer.is_capability_installed("test-capability")

		finally:
			await client.close()


if __name__ == '__main__':
	pytest.main([__file__, '-v'])
