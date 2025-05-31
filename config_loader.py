#!/usr/bin/env python3
"""
Configuration loader for Virtual Try-On CLI
Supports YAML configuration files and command-line overrides
"""

import yaml
import os
import argparse
from typing import Dict, Any, Optional

class ConfigLoader:
    """Load and manage configuration for virtual try-on"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize config loader"""
        self.config_file = config_file or "config_tryon.yaml"
        self.config = self.load_default_config()
        
        # Load from file if it exists
        if os.path.exists(self.config_file):
            self.load_from_file(self.config_file)
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "model": {
                "path": "franciszzj/Leffa",
                "viton_type": "hd",
                "device": "auto"
            },
            "inference": {
                "steps": 20,
                "cfg": 2.5,
                "seed": None
            },
            "processing": {
                "skip_pose": False,
                "target_size": 768
            },
            "paths": {
                "user_image": "",
                "garment_image": "",
                "mask_image": "",
                "output": "output_tryon.png"
            },
            "mask_urls": {
                "upper": "https://storage.googleapis.com/mask_images/upper_mask.png",
                "lower": "https://storage.googleapis.com/mask_images/lower_mask.png",
                "full": "https://storage.googleapis.com/mask_images/upper_mask.png"
            }
        }
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            
            # Merge with default config
            self.config = self.merge_configs(self.config, file_config)
            print(f"✅ Configuration loaded from: {config_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load config file {config_file}: {e}")
            print("   Using default configuration")
    
    def merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'model.path')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
    
    def apply_example_config(self, example_name: str):
        """Apply a predefined example configuration"""
        examples = self.config.get("examples", {})
        
        if example_name not in examples:
            available = list(examples.keys())
            raise ValueError(f"Example '{example_name}' not found. Available: {available}")
        
        example_config = examples[example_name]
        
        # Apply example settings
        for key, value in example_config.items():
            if key in ["viton_type"]:
                self.set(f"model.{key}", value)
            elif key in ["steps", "cfg", "seed"]:
                self.set(f"inference.{key}", value)
            elif key in ["skip_pose", "target_size"]:
                self.set(f"processing.{key}", value)
            elif key in ["user_image", "garment_image", "mask_image", "output"]:
                self.set(f"paths.{key}", value)
        
        print(f"✅ Applied example configuration: {example_name}")
    
    def get_mask_url(self, mask_type: str) -> str:
        """Get predefined mask URL for the given type"""
        mask_urls = self.config.get("mask_urls", {})
        return mask_urls.get(mask_type, "")
    
    def override_from_args(self, args: argparse.Namespace):
        """Override configuration with command-line arguments"""
        # Map command-line arguments to config paths
        arg_mappings = {
            "model_path": "model.path",
            "viton_type": "model.viton_type",
            "device": "model.device",
            "steps": "inference.steps",
            "cfg": "inference.cfg",
            "seed": "inference.seed",
            "skip_pose": "processing.skip_pose",
            "user_image": "paths.user_image",
            "garment_image": "paths.garment_image",
            "mask_image": "paths.mask_image",
            "output": "paths.output"
        }
        
        # Apply overrides
        for arg_name, config_path in arg_mappings.items():
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                if value is not None:
                    self.set(config_path, value)
    
    def save_to_file(self, output_file: str):
        """Save current configuration to YAML file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            print(f"✅ Configuration saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    def print_config(self):
        """Print current configuration"""
        print("\n📋 Current Configuration:")
        print("=" * 40)
        
        def print_section(section_name, section_data, indent=0):
            prefix = "  " * indent
            print(f"{prefix}{section_name}:")
            
            for key, value in section_data.items():
                if isinstance(value, dict):
                    print_section(key, value, indent + 1)
                else:
                    print(f"{prefix}  {key}: {value}")
        
        for section_name, section_data in self.config.items():
            if isinstance(section_data, dict) and section_name not in ["examples", "advanced"]:
                print_section(section_name, section_data)
        
        print("=" * 40)
    
    def validate_config(self) -> bool:
        """Validate the current configuration"""
        errors = []
        
        # Check required paths
        required_paths = ["user_image", "garment_image", "mask_image"]
        for path_key in required_paths:
            path_value = self.get(f"paths.{path_key}")
            if not path_value:
                errors.append(f"Missing required path: {path_key}")
        
        # Check inference parameters
        steps = self.get("inference.steps")
        if not isinstance(steps, int) or steps < 1 or steps > 100:
            errors.append("inference.steps must be an integer between 1 and 100")
        
        cfg = self.get("inference.cfg")
        if not isinstance(cfg, (int, float)) or cfg < 0.1 or cfg > 20.0:
            errors.append("inference.cfg must be a number between 0.1 and 20.0")
        
        # Check viton_type
        viton_type = self.get("model.viton_type")
        if viton_type not in ["hd", "dc"]:
            errors.append("model.viton_type must be either 'hd' or 'dc'")
        
        if errors:
            print("❌ Configuration validation errors:")
            for error in errors:
                print(f"   • {error}")
            return False
        
        print("✅ Configuration validation passed")
        return True


def create_sample_config():
    """Create a sample configuration file"""
    config = ConfigLoader()
    config.save_to_file("config_tryon_sample.yaml")
    print("📝 Sample configuration file created: config_tryon_sample.yaml")


if __name__ == "__main__":
    # Test the configuration loader
    config = ConfigLoader()
    config.print_config()
    
    # Test validation
    config.validate_config()
    
    # Create sample config
    create_sample_config()
