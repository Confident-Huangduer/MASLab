"""
MASLab integration for the maas multi-agent framework.

This module sets up the import paths so that the maas package can be imported
correctly within the MASLab structure without modifying all internal imports.
"""

import sys
import os
from pathlib import Path

# Add the maas package to Python path so internal absolute imports work
maas_package_dir = Path(__file__).parent
if str(maas_package_dir) not in sys.path:
    sys.path.insert(0, str(maas_package_dir))

# Apply config patch BEFORE importing any maas modules
from .maas_config_patch import apply_config_patch
apply_config_patch()

# Import the main class
from .maas_main import MAAS_Main

__all__ = ['MAAS_Main'] 