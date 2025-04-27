"""
Dataset generator for trust research.
"""
import os
import json
import random
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

from src.utils.config import ensure_output_dirs, DatasetConfig, GEMINI_MODELS
from src.generators.conversation import ConversationGenerator

class DatasetGenerator:
    """Generator for complete trust dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        """Initialize dataset generator.
        
        Args:
            config: Dataset configuration
        """
        self.config = config or DatasetConfig()
        ensure_output_dirs(self.config)
        
    def generate_conversation(self, model_name: str, scenario: str, user_id: str, conv_index: int) -> str:
        """Generate a single conversation using specified model.
        
        Args:
            model_name: LLM model to use
            scenario: Conversation scenario
            user_id: User identifier
            conv_index: Conversation index for ID generation
            
        Returns:
            Path to saved conversation file
        """
        # Create generator
        generator = ConversationGenerator(model_name)
        
        # Generate turns count in the specified range
        min_turns, max_turns = self.config.conversation_turns_range
        num_turns = random.randint(min_turns, max_turns)
        
        # Generate conversation
        conversation = generator.generate_conversation(
            scenario=scenario,
            user_id=user_id,
            num_turns=num_turns
        )
        
        # Create ID with index for uniqueness
        conversation.metadata.conversation_id = f"conv_{conv_index:04d}"
        
        # Save conversation
        filepath = os.path.join(
            self.config.output_dir, 
            "conversations", 
            f"{conversation.metadata.conversation_id}.json"
        )
        conversation.save(filepath)
        
        print(f"Generated conversation {conversation.metadata.conversation_id} using {model_name}")
        return filepath
    
    def generate_dataset(self, model_names: List[str] = None) -> Dict[str, Any]:
        """Generate complete dataset.
        
        Args:
            model_names: List of models to use (uses all available if None)
            
        Returns:
            Dataset statistics
        """
        # Default to all models if none specified
        if not model_names:
            model_names = list(GEMINI_MODELS.keys())
        
        # Validate models
        valid_models = [m for m in model_names if m in GEMINI_MODELS]
        if not valid_models:
            raise ValueError(f"No valid models specified. Available models: {list(GEMINI_MODELS.keys())}")
        
        # Statistics for tracking
        stats = {
            "total_conversations": self.config.num_conversations,
            "models_used": valid_models,
            "conversations_per_model": {},
            "conversations_per_scenario": {},
            "filepaths": []
        }
        
        # Distribute conversations across models
        conversations_per_model = self.config.num_conversations // len(valid_models)
        remainder = self.config.num_conversations % len(valid_models)
        
        model_counts = {}
        for i, model in enumerate(valid_models):
            count = conversations_per_model
            if i < remainder:
                count += 1
            model_counts[model] = count
            stats["conversations_per_model"][model] = count
        
        # Create all user IDs in advance
        user_ids = [f"user_{i:04d}" for i in range(self.config.num_conversations)]
        
        # Distribute scenarios
        scenario_assignments = []
        for i in range(self.config.num_conversations):
            scenario = random.choice(self.config.scenarios)
            scenario_assignments.append(scenario)
            stats["conversations_per_scenario"][scenario] = stats["conversations_per_scenario"].get(scenario, 0) + 1
        
        # Generate conversations
        conv_index = 0
        for model, count in model_counts.items():
            for i in range(count):
                filepath = self.generate_conversation(
                    model_name=model,
                    scenario=scenario_assignments[conv_index],
                    user_id=user_ids[conv_index],
                    conv_index=conv_index
                )
                stats["filepaths"].append(filepath)
                conv_index += 1
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
        
        # Generate aggregated dataset information
        self._generate_dataset_info(stats)
        
        return stats
    
    def _generate_dataset_info(self, stats: Dict[str, Any]) -> None:
        """Generate aggregated dataset information.
        
        Args:
            stats: Dataset statistics
        """
        # Load all conversations
        conversations = []
        for filepath in stats["filepaths"]:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    conversations.append(json.load(f))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Create dataset info
        dataset_info = {
            "dataset_name": "Trust Research Synthetic Dataset",
            "creation_date": time.strftime("%Y-%m-%d"),
            "num_conversations": len(conversations),
            "models_used": stats["models_used"],
            "scenarios": list(stats["conversations_per_scenario"].keys()),
            "statistics": {
                "conversations_per_model": stats["conversations_per_model"],
                "conversations_per_scenario": stats["conversations_per_scenario"]
            },
            "schema_version": "1.0"
        }
        
        # Save dataset info
        info_path = os.path.join(self.config.output_dir, "dataset_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
