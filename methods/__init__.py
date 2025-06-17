import importlib
from types import ModuleType
from typing import Dict, Tuple

__all__ = ["get_method_class"]

# ---------------------------------------------------------------------------
# Lazy registry: mapping from *public* method name (argument to --method_name)
# to a tuple (module_path, class_name).  The actual import happens only when
# the user explicitly selects that method via get_method_class().  This avoids
# importing optional heavy dependencies at MASLab startup.
# ---------------------------------------------------------------------------
_method_registry: Dict[str, Tuple[str, str]] = {
    # Core methods that have minimal external deps
    "vanilla": ("methods.mas_base", "MAS"),
    "cot": ("methods.cot", "CoT"),

    # AgentVerse variants
    "agentverse": ("methods.agentverse.agentverse_main", "AgentVerse_Main"),
    "agentverse_humaneval": ("methods.agentverse.agentverse_humaneval", "AgentVerse_HumanEval"),
    "agentverse_mgsm": ("methods.agentverse.agentverse_mgsm", "AgentVerse_MGSM"),

    # Maas integration (the focus of current work)
    "maas": ("methods.maas.maas_main", "MAAS_Main"),

    # The remaining methods are kept but lazily imported.  They may rely on
    # optional third-party libraries which might not be installed in every
    # environment.  Users can install the required dependencies when they need
    # these methods.
    "llm_debate": ("methods.llm_debate", "LLM_Debate_Main"),
    "autogen": ("methods.autogen.autogen_main", "AutoGen_Main"),
    "camel": ("methods.camel", "CAMEL_Main"),
    "evomac": ("methods.evomac", "EvoMAC_Main"),
    "chatdev_srdd": ("methods.chatdev.chatdev_srdd", "ChatDev_SRDD"),
    "macnet": ("methods.macnet.macnet_main", "MacNet_Main"),
    "macnet_srdd": ("methods.macnet.macnet_srdd", "MacNet_SRDD"),
    "mad": ("methods.mad", "MAD_Main"),
    "mapcoder_humaneval": ("methods.mapcoder.mapcoder_humaneval", "MapCoder_HumanEval"),
    "mapcoder_mbpp": ("methods.mapcoder.mapcoder_mbpp", "MapCoder_MBPP"),
    # DyLAN and SelfConsistency variants
    "dylan": ("methods.dylan.dylan_main", "DyLAN_Main"),
    "dylan_humaneval": ("methods.dylan.dylan_humaneval", "DyLAN_HumanEval"),
    "dylan_math": ("methods.dylan.dylan_math", "DyLAN_MATH"),
    "dylan_mmlu": ("methods.dylan.dylan_mmlu", "DyLAN_MMLU"),
    "self_consistency": ("methods.self_consistency", "SelfConsistency"),
}


def _import_class(module_path: str, class_name: str):
    """Import *class_name* from *module_path* and return the class object."""
    module: ModuleType = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'.") from exc


def get_method_class(method_name: str, dataset_name: str = None):
    """Return the *class* corresponding to *method_name* (case-insensitive).  This
    function also performs a fuzzy match when *dataset_name* is provided, to
    pick the variant tailored for that dataset (e.g. agentverse_humaneval).
    """
    method_name = method_name.lower()
    
    # First stage: direct match or substring match within registry keys
    candidate_keys = [key for key in _method_registry if method_name in key]
    if not candidate_keys:
        raise ValueError(f"[ERROR] No method found matching '{method_name}'. Available: {_method_registry.keys()}")

    # If dataset name is provided, further filter by suffix match
    if dataset_name:
        dataset_name = dataset_name.lower()
        ds_filtered = [key for key in candidate_keys if key.split("_")[-1] in dataset_name]
        if ds_filtered:
            candidate_keys = ds_filtered

    # Use first candidate if multiple remain (warn user)
    chosen_key = candidate_keys[0]
    if len(candidate_keys) > 1:
        print(f"[WARNING] Multiple methods matched '{method_name}': {candidate_keys}. Using '{chosen_key}'.")

    module_path, class_name = _method_registry[chosen_key]

    try:
        method_cls = _import_class(module_path, class_name)
    except Exception as exc:
        raise ImportError(
            f"Failed to import method '{chosen_key}' (module '{module_path}', class '{class_name}'). "
            "Please ensure all optional dependencies are installed." ) from exc
    
    return method_cls