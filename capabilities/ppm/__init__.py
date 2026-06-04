"""
Project & Portfolio Management (PPM) capability domain.

Sub-capabilities
----------------
ppm_pac  Project Accounting
ppm_pan  Portfolio Analytics
ppm_pbl  Project Baseline Management
ppm_pps  Project Planning & Scheduling
ppm_res  Resource Management
ppm_tex  Time & Expense Management
"""

from __future__ import annotations

CAPABILITY_IDS: list[str] = [
	"ppm_pac",
	"ppm_pan",
	"ppm_pbl",
	"ppm_pps",
	"ppm_res",
	"ppm_tex",
]

__all__ = ["CAPABILITY_IDS"]
