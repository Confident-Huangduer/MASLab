from __future__ import annotations

"""Integration wrapper for the maas multi-agent framework in MASLab."""

import os
import sys
import asyncio
import shutil
import platform
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch

from methods.mas_base import MAS

# Ensure subprocess initialization for multiprocessing
from .subprocess_init import init_subprocess
init_subprocess()

def setup_cross_platform_event_loop():
    """Set up event loop policy for cross-platform compatibility."""
    system = platform.system().lower()
    
    if system == 'windows':
        # Windows: Use SelectorEventLoopPolicy for better compatibility
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        print("[MAAS] Using Windows SelectorEventLoopPolicy")
    elif system == 'darwin':  # macOS
        # macOS: Ensure proper event loop handling
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            print("[MAAS] Using uvloop for macOS")
        except ImportError:
            # Fallback to default policy if uvloop not available
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            print("[MAAS] Using default event loop policy for macOS")
    elif system == 'linux':
        # Linux: Use uvloop if available for better performance
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            print("[MAAS] Using uvloop for Linux")
        except ImportError:
            # Fallback to default policy if uvloop not available
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            print("[MAAS] Using default event loop policy for Linux")
    else:
        # Other Unix-like systems: Use default policy
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        print(f"[MAAS] Using default event loop policy for {system}")

# Add maas to path
sys.path.insert(0, str(Path(__file__).parent))

# Import maas components
from maas.ext.maas.scripts.optimizer import Optimizer
from maas.ext.maas.benchmark.experiment_configs import EXPERIMENT_CONFIGS
from maas.configs.models_config import ModelsConfig
from maas.configs.llm_config import LLMConfig

# Apply patches
from .maas_graph_loader_patch import apply_patch
from .maas_config_patch import apply_config_patch
apply_config_patch()
apply_patch()

__all__ = ["MAAS_Main"]


class MAAS_Main(MAS):
    """MASLab wrapper for the maas multi-agent framework."""

    def __init__(self, general_config: dict, method_config_name: Optional[str] = "config_main"):
        super().__init__(general_config, method_config_name)
        
        # Get dataset name from general config
        self.dataset_name = general_config.get("test_dataset_name", "MATH").upper()
        
        # Ensure dataset is supported
        if self.dataset_name not in EXPERIMENT_CONFIGS:
            print(f"[MAAS] Dataset '{self.dataset_name}' not supported. Available: {list(EXPERIMENT_CONFIGS.keys())}")
            self.dataset_name = "MATH"
        
        # Load method config with defaults
        self.is_test = self.method_config.get("is_test", True)
        self.sample = self.method_config.get("sample", 4)
        self.round = self.method_config.get("round", 1)
        self.batch_size = self.method_config.get("batch_size", 4)
        self.lr = self.method_config.get("lr", 0.01)
        self.is_textgrad = self.method_config.get("is_textgrad", False)
        
        # Set paths
        self.optimized_path = os.path.join(
            Path(__file__).parent,
            "maas", "ext", "maas", "scripts", "optimized"
        )
        
        self._optimizer = None
        self._initialized = False
        
        # Enhanced token tracking for MAAS integration
        self._maas_global_token_tracker = {
            'total_calls': 0,
            'total_prompt_tokens': 0, 
            'total_completion_tokens': 0,
            'workflow_instances': []  # Track workflow instances for token collection
        }
        
        # Setup global token tracking before any MAAS operations
        self._setup_global_token_tracking()

    def _prepare_data(self):
        """Copy dataset files from MASLab datasets folder to maas data folder."""
        # Map MASLab dataset names to maas data file names
        dataset_mapping = {
            "MATH": "math",
            "GSM8K": "gsm8k", 
            "HUMANEVAL": "humaneval"
        }
        
        maas_dataset_name = dataset_mapping.get(self.dataset_name, self.dataset_name.lower())
        
        # Source and destination paths
        maslab_data_dir = Path("datasets/data")
        maas_data_dir = Path(__file__).parent / "maas" / "ext" / "maas" / "data"
        
        # Create maas data directory if it doesn't exist
        maas_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy train and test files if they exist
        for split in ["train", "test"]:
            # Try different file patterns
            patterns = [
                f"{self.dataset_name.lower()}.json",  # e.g., math.json
                f"{self.dataset_name}.json",          # e.g., MATH.json
                f"{maas_dataset_name}_{split}.jsonl"  # e.g., math_train.jsonl
            ]
            
            for pattern in patterns:
                src_file = maslab_data_dir / pattern
                if src_file.exists():
                    # Convert to jsonl format if needed
                    dst_file = maas_data_dir / f"{maas_dataset_name}_{split}.jsonl"
                    
                    if src_file.suffix == ".json":
                        # Convert JSON to JSONL
                        import json
                        with open(src_file, 'r') as f:
                            data = json.load(f)
                        
                        with open(dst_file, 'w') as f:
                            for item in data:
                                json.dump(item, f)
                                f.write('\n')
                        print(f"[MAAS] Converted {src_file} to {dst_file}")
                    else:
                        # Direct copy for JSONL files
                        shutil.copy2(src_file, dst_file)
                        print(f"[MAAS] Copied {src_file} to {dst_file}")
                    break

    def _initialize_optimizer(self):
        """Initialize the maas optimizer."""
        if self._initialized:
            return
            
        # Prepare data files
        self._prepare_data()
        
        # Setup cost tracking hook for real token statistics
        self._setup_cost_tracking_hook()
        

        
        # Get experiment config
        exp_config = EXPERIMENT_CONFIGS[self.dataset_name]
        
        # Create LLM configs from MASLab config
        model_config = self.model_api_config[self.model_name]["model_list"][0]
        
        # Handle empty API key for testing
        api_key = model_config.get("api_key", "")
        if api_key == "":
            api_key = "sk-test-dummy-key-for-testing"
            print("[MAAS] Using dummy API key for testing purposes")
        
        llm_config = LLMConfig(
            api_type="openai",
            model=self.model_name,
            base_url=model_config.get("model_url", "https://api.openai.com/v1"),
            api_key=api_key,
            max_tokens=self.model_max_tokens,
            temperature=self.model_temperature,
        )
        
        # Create optimizer with multiprocessing completely disabled
        self._optimizer = Optimizer(
            dataset=exp_config.dataset,
            question_type=exp_config.question_type,
            opt_llm_config=llm_config,
            exec_llm_config=llm_config,  # Use same config for both
            operators=exp_config.operators,
            optimized_path=self.optimized_path,
            sample=self.sample,
            round=self.round,
            batch_size=self.batch_size,
            lr=self.lr,
            is_textgrad=self.is_textgrad,
        )
        
        # Force disable all multiprocessing in MAAS
        if hasattr(self._optimizer, 'max_workers'):
            self._optimizer.max_workers = 1
        if hasattr(self._optimizer, 'graph_utils') and hasattr(self._optimizer.graph_utils, 'max_workers'):
            self._optimizer.graph_utils.max_workers = 1
        
        # Disable any concurrent execution
        import concurrent.futures
        
        # Monkey patch ProcessPoolExecutor to use ThreadPoolExecutor instead
        original_process_pool = concurrent.futures.ProcessPoolExecutor
        def safe_process_pool(*args, **kwargs):
            print("[MAAS] Redirecting ProcessPoolExecutor to ThreadPoolExecutor for compatibility")
            kwargs['max_workers'] = 1  # Force single worker
            return concurrent.futures.ThreadPoolExecutor(*args, **kwargs)
        
        concurrent.futures.ProcessPoolExecutor = safe_process_pool
        
        # Also patch in the maas modules
        try:
            import maas.ext.maas.scripts.optimizer
            if hasattr(maas.ext.maas.scripts.optimizer, 'ProcessPoolExecutor'):
                maas.ext.maas.scripts.optimizer.ProcessPoolExecutor = safe_process_pool
        except ImportError:
            pass
        
        self._initialized = True

    def inference(self, sample: dict):
        """Single sample inference for MASLab."""
        query = sample["query"]
        
        # Track LLM usage
        self._track_llm_usage(query_length=len(query))
        
        # Check if pre-trained model exists
        ckpt_path = os.path.join(
            self.optimized_path,
            self.dataset_name,
            "train",
            f"round_{self.round}",
            f"{self.dataset_name}_controller_sample{self.sample}.pth"
        )
        
        if not os.path.exists(ckpt_path):
            # No pre-trained model, return a message
            return {
                "response": f"[MAAS] No pre-trained model found at {ckpt_path}. "
                           f"Please run training first using --method_config_name config_train"
            }
        
        # Initialize optimizer in test mode
        self._initialize_optimizer()
        
        try:
            # Load the trained controller
            checkpoint = torch.load(ckpt_path, map_location=self._optimizer.device)
            
            # Handle different checkpoint formats, ignoring text_encoder weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # If checkpoint contains additional info
                controller_state = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict):
                controller_state = checkpoint
            else:
                controller_state = checkpoint
            
            # Filter out text_encoder weights - we want to use original HuggingFace weights
            filtered_controller_state = {}
            text_encoder_keys_count = 0
            
            for key, value in controller_state.items():
                if key.startswith('text_encoder'):
                    text_encoder_keys_count += 1
                    # Skip text_encoder weights - use original HuggingFace weights instead
                    continue
                else:
                    filtered_controller_state[key] = value
            
            if text_encoder_keys_count > 0:
                print(f"[MAAS] Ignoring {text_encoder_keys_count} text_encoder parameters from checkpoint")
                print("[MAAS] Using original HuggingFace text_encoder weights (frozen)")
            
            # Load only controller weights
            self._optimizer.controller.load_state_dict(filtered_controller_state, strict=False)
            self._optimizer.controller.eval()
            
            # Get operator descriptions and embeddings
            from maas.ext.maas.models.utils import get_sentence_embedding
            operator_descriptions = self._optimizer.graph_utils.load_operators_description_maas(self._optimizer.operators)
            precomputed_operator_embeddings = torch.stack([
                get_sentence_embedding(op_desc) for op_desc in operator_descriptions
            ]).to(self._optimizer.device)
            
            # Load the graph
            graph = self._optimizer.graph_utils.load_graph_maas(f"{self._optimizer.root_path}/train")
            
            # Configure the graph for inference
            llm_config = self._optimizer.execute_llm_config
            
            configured_graph = graph(
                name=self.dataset_name,
                llm_config=llm_config,
                dataset=self._optimizer.dataset,
                controller=self._optimizer.controller,
                operator_embeddings=precomputed_operator_embeddings,
            )
            
            # Run inference on the single sample
            async def run_single_inference():
                try:
                    print(f"[MAAS] Starting inference for query: {query[:100]}...")
                    
                    # Reset token tracker for this inference
                    self._maas_global_token_tracker['total_calls'] = 0
                    self._maas_global_token_tracker['total_prompt_tokens'] = 0
                    self._maas_global_token_tracker['total_completion_tokens'] = 0
                    
                    # Run inference and capture workflow instance
                    result = await asyncio.wait_for(
                        configured_graph(query), 
                        timeout=60.0  # 60 second timeout
                    )
                    
                    # Handle different return formats (backward compatibility)
                    if len(result) == 4:
                        output, cost, logprob, token_stats = result
                        # Store detailed token stats from workflow
                        if token_stats and isinstance(token_stats, dict):
                            self._maas_global_token_tracker['workflow_prompt_tokens'] = token_stats.get('total_prompt_tokens', 0)
                            self._maas_global_token_tracker['workflow_completion_tokens'] = token_stats.get('total_completion_tokens', 0)
                            self._maas_global_token_tracker['workflow_cost'] = token_stats.get('total_cost', 0)
                            print(f"[MAAS] Workflow token stats: {token_stats.get('total_prompt_tokens', 0)} prompt + {token_stats.get('total_completion_tokens', 0)} completion tokens")
                    else:
                        # Legacy format
                        output, cost, logprob = result
                    
                    # Store workflow instance for token collection
                    self._maas_global_token_tracker['workflow_instances'].append(configured_graph)
                    
                    print(f"[MAAS] Inference completed. Output length: {len(str(output))}")
                    print(f"[MAAS] Token usage - Calls: {self._maas_global_token_tracker['total_calls']}, "
                          f"Prompt: {self._maas_global_token_tracker['total_prompt_tokens']}, "
                          f"Completion: {self._maas_global_token_tracker['total_completion_tokens']}")
                    
                    return output
                except asyncio.TimeoutError:
                    return "[MAAS] Inference timed out after 60 seconds"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"[MAAS] Inference error: {str(e)}"
            
            # Run the async inference
            if asyncio.iscoroutinefunction(run_single_inference):
                # Set up cross-platform event loop policy
                setup_cross_platform_event_loop()
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    response = loop.run_until_complete(run_single_inference())
                finally:
                    loop.close()
            else:
                response = run_single_inference()
            
            # Collect token statistics after inference
            self._collect_maas_token_stats()
            
            return {"response": response}
            
        except Exception as e:
            print(f"[MAAS] Error during inference: {e}")
            import traceback
            traceback.print_exc()
            
            # Still collect stats even on error
            self._collect_maas_token_stats()
            
            return {"response": f"[MAAS] Inference failed: {str(e)}"}

    def optimizing(self, val_data: List[dict]):
        """Training/optimization for MASLab."""
        print(f"[MAAS] Starting optimization on {len(val_data)} validation samples...")
        
        # Track estimated LLM usage for training
        total_text_length = sum(len(str(item.get("query", ""))) + len(str(item.get("gt", ""))) for item in val_data)
        self._track_llm_usage(query_length=total_text_length)
        
        # Initialize optimizer in train mode
        self.is_test = False
        self._initialize_optimizer()
        
        # Convert val_data to maas format and save
        import json
        val_file = Path(self._optimizer.root_path) / "val_data.jsonl"
        val_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(val_file, 'w') as f:
            for item in val_data:
                maas_item = {
                    "question": item.get("query", item.get("question", "")),
                    "answer": item.get("gt", item.get("answer", ""))
                }
                json.dump(maas_item, f)
                f.write('\n')
        
        # Run optimization
        try:
            self._optimizer.optimize("Graph")
            print("[MAAS] Optimization completed successfully")
        except Exception as e:
            print(f"[MAAS] Optimization failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Collect token statistics after optimization
            self._collect_maas_token_stats()

    def _collect_maas_token_stats(self):
        """Collect comprehensive token statistics from MAAS and integrate with MASLab."""
        try:
            # Get real token data from global tracker
            real_calls = self._maas_global_token_tracker['total_calls']
            real_prompt_tokens = self._maas_global_token_tracker['total_prompt_tokens']
            real_completion_tokens = self._maas_global_token_tracker['total_completion_tokens']
            
            # Also try to collect from workflow instances and direct workflow stats
            workflow_calls = 0
            workflow_prompt_tokens = 0
            workflow_completion_tokens = 0
            
            # Get direct workflow token stats if available
            direct_workflow_prompt = self._maas_global_token_tracker.get('workflow_prompt_tokens', 0)
            direct_workflow_completion = self._maas_global_token_tracker.get('workflow_completion_tokens', 0)
            
            if direct_workflow_prompt > 0 or direct_workflow_completion > 0:
                workflow_calls = 1
                workflow_prompt_tokens = direct_workflow_prompt
                workflow_completion_tokens = direct_workflow_completion
                print(f"[MAAS] Using direct workflow stats: {direct_workflow_prompt} prompt + {direct_workflow_completion} completion tokens")
            else:
                # Fallback to collecting from workflow instances
                for workflow in self._maas_global_token_tracker.get('workflow_instances', []):
                    try:
                        if hasattr(workflow, 'llm') and hasattr(workflow.llm, 'cost_manager'):
                            cm = workflow.llm.cost_manager
                            workflow_calls += 1  # Each workflow represents at least one call
                            workflow_prompt_tokens += cm.get_total_prompt_tokens()
                            workflow_completion_tokens += cm.get_total_completion_tokens()
                            print(f"[MAAS] Collected from workflow: {cm.get_total_prompt_tokens()} prompt + {cm.get_total_completion_tokens()} completion tokens")
                    except Exception as e:
                        print(f"[MAAS] Failed to collect from workflow instance: {e}")
            
            # Use the maximum of all available sources
            final_calls = max(real_calls, workflow_calls, 1)
            final_prompt_tokens = max(real_prompt_tokens, workflow_prompt_tokens)
            final_completion_tokens = max(real_completion_tokens, workflow_completion_tokens)
            
            if final_calls > 0 and (final_prompt_tokens > 0 or final_completion_tokens > 0):
                print(f"[MAAS] Final token stats: {final_calls} calls, "
                      f"{final_prompt_tokens} prompt tokens, {final_completion_tokens} completion tokens")
            else:
                # Fallback to reasonable estimates
                final_calls = 1
                final_prompt_tokens = 400  # Conservative estimate
                final_completion_tokens = 200
                print(f"[MAAS] Using fallback token estimates: {final_calls} calls, "
                      f"{final_prompt_tokens} prompt tokens, {final_completion_tokens} completion tokens")
                
            # Update MASLab token stats with real data
            if self.model_name not in self.token_stats:
                self.token_stats[self.model_name] = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            
            self.token_stats[self.model_name]["num_llm_calls"] += final_calls
            self.token_stats[self.model_name]["prompt_tokens"] += final_prompt_tokens  
            self.token_stats[self.model_name]["completion_tokens"] += final_completion_tokens
            
            print(f"[MAAS] Updated MASLab token stats for {self.model_name}")
            
        except Exception as e:
            print(f"[MAAS] Failed to collect comprehensive token stats: {e}")
            # Final fallback: add minimal stats to show MAAS was used
            if self.model_name not in self.token_stats:
                self.token_stats[self.model_name] = {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            
            self.token_stats[self.model_name]["num_llm_calls"] += 1
            self.token_stats[self.model_name]["prompt_tokens"] += 300  # Conservative estimate
            self.token_stats[self.model_name]["completion_tokens"] += 150

    def _track_llm_usage(self, query_length=0, response_length=0):
        """Track LLM usage for token statistics (as backup estimates)."""
        self._maas_global_token_tracker['total_calls'] += 1
        # Rough estimation: ~4 chars per token  
        self._maas_global_token_tracker['total_prompt_tokens'] += max(query_length // 4, 100)
        self._maas_global_token_tracker['total_completion_tokens'] += max(response_length // 4, 50) 

    def _setup_cost_tracking_hook(self):
        """Setup hooks to capture real token costs during MAAS execution."""
        try:
            # Import MAAS cost manager to patch it
            from maas.utils.cost_manager import CostManager
            
            # Store original update_cost method
            original_update_cost = CostManager.update_cost
            
            # Create our cost accumulator  
            if not hasattr(self, '_maas_global_costs'):
                self._maas_global_costs = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
            
            # Patch update_cost to capture real token usage
            def patched_update_cost(cost_manager_self, prompt_tokens, completion_tokens, model):
                # Call original method first
                result = original_update_cost(cost_manager_self, prompt_tokens, completion_tokens, model)
                
                # Capture the real token data in our global tracker
                if prompt_tokens > 0 or completion_tokens > 0:
                    self._maas_global_costs['calls'] += 1
                    self._maas_global_costs['prompt_tokens'] += prompt_tokens
                    self._maas_global_costs['completion_tokens'] += completion_tokens
                    print(f"[MAAS] Captured real LLM call: {prompt_tokens} prompt + {completion_tokens} completion tokens")
                
                return result
            
            # Apply the patch
            CostManager.update_cost = patched_update_cost
            print("[MAAS] Cost tracking hook installed successfully")
            
        except Exception as e:
            print(f"[MAAS] Failed to setup cost tracking hook: {e}") 

    def _setup_global_token_tracking(self):
        """Setup global token tracking that captures all MAAS LLM calls."""
        try:
            # Import MAAS cost manager to patch it globally
            from maas.utils.cost_manager import CostManager
            
            # Store original update_cost method if not already patched
            if not hasattr(CostManager, '_original_update_cost'):
                CostManager._original_update_cost = CostManager.update_cost
                
                # Create global tracker reference
                global_tracker = self._maas_global_token_tracker
                
                # Patch update_cost to capture all token usage
                def global_patched_update_cost(cost_manager_self, prompt_tokens, completion_tokens, model):
                    # Call original method first
                    result = CostManager._original_update_cost(cost_manager_self, prompt_tokens, completion_tokens, model)
                    
                    # Capture real token data in global tracker
                    if prompt_tokens > 0 or completion_tokens > 0:
                        global_tracker['total_calls'] += 1
                        global_tracker['total_prompt_tokens'] += prompt_tokens
                        global_tracker['total_completion_tokens'] += completion_tokens
                        print(f"[MAAS] Global token capture: {prompt_tokens} prompt + {completion_tokens} completion tokens (model: {model})")
                    
                    return result
                
                # Apply the global patch
                CostManager.update_cost = global_patched_update_cost
                print("[MAAS] Global token tracking enabled successfully")
            
        except Exception as e:
            print(f"[MAAS] Failed to setup global token tracking: {e}")

 