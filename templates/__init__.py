#!/usr/bin/env python3
"""
APG Project Templates and Scaffolding
=====================================

Provides ready-to-use project templates and scaffolding tools for common
APG application patterns and use cases.
"""

from .template_manager import TemplateManager
from .project_scaffolder import ProjectScaffolder
from .template_types import TemplateType, ProjectConfig

__all__ = [
    'TemplateManager',
    'ProjectScaffolder', 
    'TemplateType',
    'ProjectConfig'
]

__version__ = "1.0.0"
__author__ = "APG Development Team"