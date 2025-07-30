"""
Computer Vision & Visual Intelligence - Blueprint Module

Flask-AppBuilder blueprint integration for APG platform providing complete
capability registration, user interface, and platform integration.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

from .blueprint import (
	ComputerVisionCapabilityBlueprint,
	ComputerVisionMiddleware,
	register_computer_vision_capability
)

__all__ = [
	"ComputerVisionCapabilityBlueprint",
	"ComputerVisionMiddleware", 
	"register_computer_vision_capability"
]