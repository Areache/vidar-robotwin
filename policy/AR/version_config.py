"""
Version configuration loader.

Loads version configurations from JSON files and provides utilities
for applying version parameters.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from .version_registry import VersionRegistry, get_registry, load_default_versions

logger = logging.getLogger(__name__)


class VersionConfig:
    """Configuration manager for versions."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize version configuration.
        
        Args:
            config_file: Path to version configuration JSON file.
                        If None, uses default versions.
        """
        self.registry = get_registry()
        
        if config_file and Path(config_file).exists():
            self.registry.load_from_file(config_file)
        else:
            # Load default versions
            load_default_versions()
            logger.info("Loaded default version definitions")
    
    def get_version_params(self, version_name: str) -> Dict[str, Any]:
        """
        Get all parameters for a version.
        
        Returns:
            Dictionary with 'robotwin_params' and 'vidar_params' keys
        """
        version = self.registry.get_version(version_name)
        if not version:
            raise ValueError(f"Version not found: {version_name}")
        
        return {
            'robotwin_params': version.robotwin_params.copy(),
            'vidar_params': version.vidar_params.copy(),
            'use_subgoals': version.use_subgoals
        }
    
    def apply_to_usr_args(self, usr_args: Dict[str, Any], version_name: str) -> Dict[str, Any]:
        """
        Apply version parameters to usr_args dictionary.
        
        Args:
            usr_args: Original user arguments
            version_name: Version to apply
            
        Returns:
            Updated usr_args with version parameters applied
        """
        params = self.get_version_params(version_name)
        
        # Create a copy to avoid modifying original
        updated_args = usr_args.copy()
        
        # Apply robotwin parameters
        # 注意：保持与原有代码的兼容性，use_libero_subgoal 可能是字符串类型
        robotwin_params = params['robotwin_params'].copy()
        
        # 如果原有参数是字符串类型，保持字符串类型以兼容原有代码
        # 检查原有 use_libero_subgoal 的类型
        original_use_subgoal = updated_args.get("use_libero_subgoal")
        if isinstance(original_use_subgoal, str):
            # 原有代码使用字符串，转换为字符串以保持兼容
            if "use_libero_subgoal" in robotwin_params:
                robotwin_params["use_libero_subgoal"] = str(robotwin_params["use_libero_subgoal"]).lower()
        
        updated_args.update(robotwin_params)
        
        # Store vidar params for later use
        updated_args['_version_vidar_params'] = params['vidar_params']
        updated_args['_version_name'] = version_name
        
        return updated_args
    
    def list_available_versions(self) -> list[str]:
        """List all available version names."""
        return self.registry.list_versions()


def load_version_config(config_file: Optional[str] = None) -> VersionConfig:
    """
    Convenience function to load version configuration.
    
    Args:
        config_file: Path to version config file. If None, uses default.
        
    Returns:
        VersionConfig instance
    """
    return VersionConfig(config_file)

