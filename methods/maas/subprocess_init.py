"""
Subprocess initialization for MAAS to handle multiprocessing scenarios.
This ensures proper configuration in each worker process.
"""

import os
import sys
from pathlib import Path

def init_subprocess():
    """Initialize MAAS configuration in subprocess."""
    try:
        # Add maas to Python path
        maas_root = Path(__file__).parent
        if str(maas_root) not in sys.path:
            sys.path.insert(0, str(maas_root))
        
        # Set configuration environment variables
        config_dir = maas_root / "config"
        config_file = config_dir / "config2.yaml"
        
        os.environ["METAGPT_CONFIG"] = str(config_file)
        os.environ["MAAS_CONFIG"] = str(config_file)
        os.environ["CONFIG_ROOT"] = str(config_dir)
        
        print(f"[MAAS Subprocess] Initialized config: {config_file}")
        
        # Apply configuration patch - use absolute import
        try:
            import maas_config_patch
            maas_config_patch.apply_config_patch()
        except ImportError:
            # Alternative import method
            import importlib.util
            patch_file = maas_root / "maas_config_patch.py"
            spec = importlib.util.spec_from_file_location("maas_config_patch", patch_file)
            patch_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch_module)
            patch_module.apply_config_patch()
        
        # Apply graph loader patch
        try:
            import maas_graph_loader_patch
            maas_graph_loader_patch.apply_patch()
        except ImportError:
            # Alternative import method
            import importlib.util
            patch_file = maas_root / "maas_graph_loader_patch.py"
            spec = importlib.util.spec_from_file_location("maas_graph_loader_patch", patch_file)
            patch_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patch_module)
            patch_module.apply_patch()
        
        print("[MAAS Subprocess] All patches applied successfully")
        
    except Exception as e:
        print(f"[MAAS Subprocess] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()

def get_maas_initializer():
    """Return the subprocess initializer function for multiprocessing."""
    return init_subprocess

# Auto-initialize when imported
if __name__ != "__main__":
    init_subprocess() 