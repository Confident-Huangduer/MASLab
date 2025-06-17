"""
Patch for maas config to provide default LLM configuration for MASLab integration.
"""

import os
import sys
from pathlib import Path

def apply_config_patch():
    """Set up maas to use our config directory and prevent multiprocessing issues."""
    try:
        # Set environment variable to point to our config directory  
        config_dir = Path(__file__).parent / "config"
        config_file = config_dir / "config2.yaml"
        
        # Set multiple environment variables that MAAS might check
        os.environ["METAGPT_CONFIG"] = str(config_file)
        os.environ["MAAS_CONFIG"] = str(config_file)
        os.environ["CONFIG_ROOT"] = str(config_dir)
        
        # Ensure config file exists
        if not config_file.exists():
            print(f"[MAAS Config Patch] Config file not found: {config_file}")
            return
            
        print(f"[MAAS Config Patch] Set config file to: {config_file}")
        
        # Import and patch MAAS config
        try:
            import maas.const as maas_const
            maas_const.CONFIG_ROOT = config_dir
            print(f"[MAAS Config Patch] Set MAAS CONFIG_ROOT to: {config_dir}")
        except ImportError:
            print("[MAAS Config Patch] Could not import maas.const, skipping CONFIG_ROOT setup")
        
        # Patch the Config class to be more robust
        try:
            from maas import config2
            
            # Store original method
            if not hasattr(config2.Config, '_original_default'):
                config2.Config._original_default = config2.Config.default
            
            def patched_default(cls):
                try:
                    # Try original method first
                    result = config2.Config._original_default()
                    print("[MAAS Config Patch] Using original config successfully")
                    return result
                except Exception as e:
                    print(f"[MAAS Config Patch] Original config failed ({e}), using fallback")
                    
                    # Create minimal valid config as fallback
                    from maas.config2 import LLMConfig
                    fallback_config = config2.Config(
                        llm=LLMConfig(
                            api_type="openai",
                            model="gpt-4o-mini",
                            api_key="placeholder",
                            base_url="https://api.openai.com/v1",
                            max_tokens=2048,
                            temperature=0.5
                        ),
                        proxy="",
                        repair_llm_output=False,
                        prompt_schema="json",
                        language="English"
                    )
                    print("[MAAS Config Patch] Created fallback config")
                    return fallback_config
            
            # Apply the patch
            config2.Config.default = classmethod(patched_default)
            
            # Reinitialize global config if it exists and replace module-level config
            try:
                new_config = config2.Config.default()
                config2.config = new_config
                
                # Also patch the module-level global config loading to prevent issues
                import maas.config2
                maas.config2.config = new_config
                
                print("[MAAS Config Patch] Successfully reinitialized global config")
            except Exception as e:
                print(f"[MAAS Config Patch] Failed to reinitialize global config: {e}")
                
                # If all else fails, create a minimal config directly
                try:
                    from maas.config2 import LLMConfig
                    minimal_config = config2.Config(
                        llm=LLMConfig(
                            api_type="openai",
                            model="gpt-4o-mini",
                            api_key="placeholder",
                            base_url="https://api.openai.com/v1",
                            max_tokens=2048,
                            temperature=0.5
                        )
                    )
                    config2.config = minimal_config
                    maas.config2.config = minimal_config
                    print("[MAAS Config Patch] Set minimal fallback config")
                except Exception as e2:
                    print(f"[MAAS Config Patch] Failed to set minimal config: {e2}")
            
        except ImportError as e:
            print(f"[MAAS Config Patch] Could not import maas.config2: {e}")
        except Exception as e:
            print(f"[MAAS Config Patch] Error applying config patch: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"[MAAS Config Patch] Failed to apply config patch: {e}")
        import traceback
        traceback.print_exc()


# Apply patch when module is imported
apply_config_patch() 