"""
Version Registry for managing subgoal implementation versions.

This module provides a registry pattern for managing different versions of
the subgoal implementation, allowing easy switching and comparison.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class VersionMetadata:
    """Metadata for a version definition."""
    name: str
    description: str
    use_subgoals: bool
    robotwin_params: Dict[str, Any] = field(default_factory=dict)
    vidar_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionMetadata':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            use_subgoals=data.get('use_subgoals', False),
            robotwin_params=data.get('robotwin_params', {}),
            vidar_params=data.get('vidar_params', {})
        )


class VersionRegistry:
    """
    Registry for managing subgoal implementation versions.
    
    Usage:
        registry = VersionRegistry()
        registry.load_from_file("versions/subgoal_versions.json")
        version = registry.get_version("v0_original")
    """
    
    def __init__(self):
        self._versions: Dict[str, VersionMetadata] = {}
    
    def register_version(self, version: VersionMetadata):
        """Register a version."""
        self._versions[version.name] = version
        logger.info(f"Registered version: {version.name} - {version.description}")
    
    def get_version(self, name: str) -> Optional[VersionMetadata]:
        """Get a version by name."""
        return self._versions.get(name)
    
    def list_versions(self) -> list[str]:
        """List all registered version names."""
        return list(self._versions.keys())
    
    def load_from_file(self, filepath: str):
        """Load versions from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Version file not found: {filepath}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        versions = data.get('versions', {})
        for name, version_data in versions.items():
            version = VersionMetadata.from_dict(version_data)
            self.register_version(version)
        
        logger.info(f"Loaded {len(versions)} versions from {filepath}")
    
    def load_from_dict(self, data: Dict[str, Any]):
        """Load versions from a dictionary."""
        versions = data.get('versions', {})
        for name, version_data in versions.items():
            version = VersionMetadata.from_dict(version_data)
            self.register_version(version)
    
    def get_robotwin_params(self, version_name: str) -> Dict[str, Any]:
        """Get robotwin parameters for a version."""
        version = self.get_version(version_name)
        if version:
            return version.robotwin_params.copy()
        return {}
    
    def get_vidar_params(self, version_name: str) -> Dict[str, Any]:
        """Get vidar parameters for a version."""
        version = self.get_version(version_name)
        if version:
            return version.vidar_params.copy()
        return {}


# Global registry instance
_global_registry = VersionRegistry()


def get_registry() -> VersionRegistry:
    """Get the global registry instance."""
    return _global_registry


def load_default_versions():
    """Load default version definitions."""
    registry = get_registry()
    
    # Define default versions
    default_versions = {
        "versions": {
            "v0_original": {
                "name": "v0_original",
                "description": "Original implementation without subgoal support",
                "use_subgoals": False,
                "robotwin_params": {
                    "use_libero_subgoal": False
                },
                "vidar_params": {
                    "subgoal_frames": None,
                    "subgoal_guidance_scale": 0.0
                }
            },
            "v1_subgoal": {
                "name": "v1_subgoal",
                "description": "Current implementation with LIBERO subgoal support (HTTP server mode, same as original)",
                "use_subgoals": True,
                "robotwin_params": {
                    "use_libero_subgoal": True,
                    "libero_use_direct_model": False,
                    "subgoal_interval": 8
                },
                "vidar_params": {
                    "subgoal_guidance_scale": 0.5
                }
            },
            "v2_mpc": {
                "name": "v2_mpc",
                "description": "V2 implementation with Model Predictive Control (MPC) for action optimization (no subgoals)",
                "use_subgoals": False,
                "robotwin_params": {
                    "use_libero_subgoal": False,
                    "libero_use_direct_model": False,
                    "subgoal_interval": 8,
                    "use_mpc": True,
                    "mpc_num_candidates": 5,  # 降低默认值以加快初步验证
                    "mpc_cost_weights": {"task": 1.0, "ctrl": 0.1, "reach": 0.5}
                },
                "vidar_params": {
                    "subgoal_guidance_scale": 0.0,
                    "use_mpc": True,
                    "mpc_num_candidates": 5,  # 降低默认值以加快初步验证
                    "mpc_cost_weights": {"task": 1.0, "ctrl": 0.1, "reach": 0.5}
                }
            },
            "df": {
                "name": "df",
                "description": "Diffusion forcing version - isolated implementation for inference-time diffusion forcing",
                "use_subgoals": False,
                "robotwin_params": {
                    "use_libero_subgoal": False,
                    "libero_use_direct_model": False,
                    "subgoal_interval": 8,
                    "use_diffusion_forcing": True
                },
                "vidar_params": {
                    "subgoal_frames": None,
                    "subgoal_guidance_scale": 0.0,
                    "use_diffusion_forcing": True
                }
            }
        }
    }
    
    registry.load_from_dict(default_versions)
    return registry

