#!/usr/bin/env python
"""
Verification script for testing both subgoal implementation versions.

This script tests v0_original (without subgoals) and v1_subgoal (with subgoals)
to ensure both versions work correctly.

Usage:
    python verify_versions.py [--version VERSION] [--task TASK_CONFIG]
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))

# Import with relative imports
from .version_config import load_version_config
from .version_registry import get_registry, load_default_versions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_version(version_name: str, task_config: str = "hd_clean") -> dict:
    """
    Verify a single version by checking its configuration.
    
    Args:
        version_name: Name of the version to verify
        task_config: Task configuration to use
        
    Returns:
        Dictionary with verification results
    """
    logger.info(f"Verifying version: {version_name}")
    
    try:
        # Load version configuration
        version_config = load_version_config()
        
        # Get version metadata
        registry = get_registry()
        version = registry.get_version(version_name)
        
        if not version:
            return {
                "version": version_name,
                "status": "FAILED",
                "error": f"Version {version_name} not found"
            }
        
        # Get version parameters
        params = version_config.get_version_params(version_name)
        
        # Verify parameters
        checks = {
            "version_exists": True,
            "has_robotwin_params": "robotwin_params" in params,
            "has_vidar_params": "vidar_params" in params,
            "use_subgoals_flag": params.get("use_subgoals") == version.use_subgoals,
        }
        
        # Version-specific checks
        if version_name == "v0_original":
            checks["subgoal_disabled"] = params["robotwin_params"].get("use_libero_subgoal") == False
            checks["guidance_scale_zero"] = params["vidar_params"].get("subgoal_guidance_scale") == 0.0
        elif version_name == "v1_subgoal":
            checks["subgoal_enabled"] = params["robotwin_params"].get("use_libero_subgoal") == True
            checks["guidance_scale_set"] = params["vidar_params"].get("subgoal_guidance_scale", 0) > 0
        
        all_passed = all(checks.values())
        
        return {
            "version": version_name,
            "description": version.description,
            "status": "PASSED" if all_passed else "FAILED",
            "checks": checks,
            "params": params
        }
        
    except Exception as e:
        logger.error(f"Error verifying version {version_name}: {e}", exc_info=True)
        return {
            "version": version_name,
            "status": "ERROR",
            "error": str(e)
        }


def verify_both_versions(task_config: str = "hd_clean") -> dict:
    """
    Verify both v0_original and v1_subgoal versions.
    
    Args:
        task_config: Task configuration to use
        
    Returns:
        Dictionary with verification results for both versions
    """
    logger.info("="*80)
    logger.info("Verifying Both Versions")
    logger.info("="*80)
    
    # Load default versions
    load_default_versions()
    
    results = {
        "v0_original": verify_version("v0_original", task_config),
        "v1_subgoal": verify_version("v1_subgoal", task_config)
    }
    
    # Summary
    v0_status = results["v0_original"]["status"]
    v1_status = results["v1_subgoal"]["status"]
    
    logger.info("")
    logger.info("="*80)
    logger.info("Verification Summary")
    logger.info("="*80)
    logger.info(f"v0_original: {v0_status}")
    logger.info(f"v1_subgoal: {v1_status}")
    logger.info("="*80)
    
    return results


def print_detailed_results(results: dict):
    """Print detailed verification results."""
    print("\n" + "="*80)
    print("Detailed Verification Results")
    print("="*80)
    
    for version_name, result in results.items():
        print(f"\n{version_name}:")
        print(f"  Description: {result.get('description', 'N/A')}")
        print(f"  Status: {result.get('status', 'UNKNOWN')}")
        
        if "checks" in result:
            print("  Checks:")
            for check_name, check_result in result["checks"].items():
                status = "✓" if check_result else "✗"
                print(f"    {status} {check_name}: {check_result}")
        
        if "error" in result:
            print(f"  Error: {result['error']}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Verify subgoal implementation versions"
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v0_original", "v1_subgoal", "both"],
        default="both",
        help="Version to verify (default: both)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="hd_clean",
        help="Task configuration (default: hd_clean)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify versions
    if args.version == "both":
        results = verify_both_versions(args.task)
    else:
        load_default_versions()
        results = {args.version: verify_version(args.version, args.task)}
    
    # Print detailed results
    print_detailed_results(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    
    # Exit with error code if any version failed
    all_passed = all(
        r.get("status") == "PASSED" 
        for r in results.values()
    )
    
    if not all_passed:
        logger.error("Some versions failed verification!")
        sys.exit(1)
    else:
        logger.info("All versions passed verification!")
        sys.exit(0)


if __name__ == "__main__":
    main()

