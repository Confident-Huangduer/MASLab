"""
Patch for maas graph loading to work correctly in MASLab.

This module patches the graph loading mechanism to handle paths correctly
when maas is integrated into MASLab.
"""

import os
import sys
import importlib.util
from pathlib import Path


def load_graph_maas_patched(self, workflows_path: str):
    """
    Patched version of load_graph_maas that correctly loads graph modules
    from file paths instead of trying to convert paths to module names.
    """
    # Construct the path to graph.py
    graph_file = os.path.join(workflows_path, "graph.py")
    
    if not os.path.exists(graph_file):
        raise ImportError(f"Graph file not found: {graph_file}")
    
    # Load the module from file
    spec = importlib.util.spec_from_file_location("graph", graph_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {graph_file}")
    
    graph_module = importlib.util.module_from_spec(spec)
    
    # Add the directory to sys.path temporarily so imports within graph.py work
    graph_dir = os.path.dirname(graph_file)
    sys.path.insert(0, graph_dir)
    
    try:
        spec.loader.exec_module(graph_module)
        graph_class = getattr(graph_module, "Workflow")
        return graph_class
    finally:
        # Remove the directory from sys.path
        if graph_dir in sys.path:
            sys.path.remove(graph_dir)


def apply_patch():
    """Apply the patch to GraphUtils.load_graph_maas method."""
    try:
        from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils
        # Replace the method with our patched version
        GraphUtils.load_graph_maas = load_graph_maas_patched
        print("[MAAS Patch] Successfully patched GraphUtils.load_graph_maas")
    except Exception as e:
        print(f"[MAAS Patch] Failed to apply patch: {e}") 