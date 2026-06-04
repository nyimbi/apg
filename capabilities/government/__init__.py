"""
Government & Public Sector (GPS) Industry Vertical

Government and public sector capabilities including citizen services,
case management, regulatory compliance, and public administration.
"""

__version__ = "1.0.0"

# Capability IDs exported for registry discovery
CAPABILITY_IDS = [
	"government_bud",  # Budget Management
	"government_cas",  # Case Management
	"government_con",  # Government Contracts & Procurement
	"government_csr",  # Citizen Services Portal
	"government_ele",  # Electoral & Civil Registration
	"government_eme",  # Emergency Management
	"government_law",  # Law Enforcement & Justice
	"government_lic",  # Licensing & Permits
	"government_per",  # Permits Management
	"government_tax",  # Tax Administration
]

__all__ = [
	"CAPABILITY_IDS",
	"bud",
	"cas",
	"con",
	"csr",
	"ele",
	"eme",
	"law",
	"lic",
	"per",
	"tax",
]
