"""APG Energy & Utilities Management capability domain.

Sub-capabilities
----------------
energy_gen  Generation Management
energy_dis  Distribution Network
energy_met  Smart Metering & AMI
energy_bil  Energy Billing & Tariffs
energy_ren  Renewable Energy
energy_grd  Grid Operations

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

__version__ = "1.0.0"

CAPABILITY_IDS: list[str] = [
	"energy_gen",
	"energy_dis",
	"energy_met",
	"energy_bil",
	"energy_ren",
	"energy_grd",
]

SUBCAPABILITIES: dict[str, str] = {
	"gen": "Generation Management",
	"dis": "Distribution Network",
	"met": "Smart Metering & AMI",
	"bil": "Energy Billing & Tariffs",
	"ren": "Renewable Energy",
	"grd": "Grid Operations",
}


def get_capability_ids() -> list[str]:
	"""Return all energy domain capability IDs."""
	return list(CAPABILITY_IDS)


def get_subcapabilities() -> dict[str, str]:
	"""Return mapping of subcapability code -> display name."""
	return dict(SUBCAPABILITIES)


__all__ = [
	"CAPABILITY_IDS",
	"SUBCAPABILITIES",
	"get_capability_ids",
	"get_subcapabilities",
]
