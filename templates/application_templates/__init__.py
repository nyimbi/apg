#!/usr/bin/env python3
"""
APG Application Templates
========================

Complete application templates for generating world-class, domain-specific
Flask-AppBuilder applications from APG source code.

Each template is a complete, working Flask-AppBuilder application that embodies
best practices for its specific domain.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json

__version__ = "1.0.0"
__author__ = "APG Development Team"

# Template categories
TEMPLATE_CATEGORIES = {
    'basic': 'Basic Applications',
    'intelligence': 'Intelligence Platforms', 
    'marketplace': 'Multi-Sided Marketplaces',
    'iot': 'IoT & Digital Twins',
    'fintech': 'Financial Technology',
    'healthcare': 'Healthcare Systems',
    'logistics': 'Supply Chain & Logistics',
    'enterprise': 'Enterprise Solutions'
}

def get_template_directory() -> Path:
    """Get the templates directory path"""
    return Path(__file__).parent

def list_available_templates() -> List[Dict[str, Any]]:
    """List all available application templates"""
    templates_dir = get_template_directory()
    templates = []
    
    for category_dir in templates_dir.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('__'):
            for template_dir in category_dir.iterdir():
                if template_dir.is_dir():
                    template_json = template_dir / 'template.json'
                    if template_json.exists():
                        try:
                            with open(template_json, 'r') as f:
                                template_info = json.load(f)
                            template_info['category'] = category_dir.name
                            template_info['template_id'] = f"{category_dir.name}/{template_dir.name}"
                            templates.append(template_info)
                        except Exception:
                            continue
    
    return templates

def get_template_by_id(template_id: str) -> Optional[Dict[str, Any]]:
    """Get template information by ID (category/name)"""
    templates_dir = get_template_directory()
    template_path = templates_dir / template_id
    
    if not template_path.exists():
        return None
    
    template_json = template_path / 'template.json'
    if not template_json.exists():
        return None
    
    try:
        with open(template_json, 'r') as f:
            template_info = json.load(f)
        template_info['template_path'] = template_path
        return template_info
    except Exception:
        return None