"""Timing utilities for profiling training pipeline.

Controlled by TIME_DEBUG environment variable.
"""

import os
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

# Check if timing debug is enabled
TIME_DEBUG_ENABLED = os.environ.get("TIME_DEBUG", "0") == "1"


class TimingStats:
    """Statistics for a timing category."""
    
    def __init__(self):
        self.times: List[float] = []
        self.count = 0
        
    def add(self, elapsed: float):
        """Add a timing measurement."""
        self.times.append(elapsed)
        self.count += 1
        
    def mean(self) -> float:
        """Get mean time."""
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)
    
    def min(self) -> float:
        """Get minimum time."""
        if not self.times:
            return 0.0
        return min(self.times)
    
    def max(self) -> float:
        """Get maximum time."""
        if not self.times:
            return 0.0
        return max(self.times)
    
    def reset(self):
        """Reset statistics."""
        self.times.clear()
        self.count = 0


class TimingContext:
    """Context manager for timing code blocks.
    
    Uses torch.cuda.Event for accurate GPU timing when available,
    falls back to time.time() for CPU operations.
    """
    
    def __init__(self, name: str, device: Optional[torch.device] = None, 
                 print_immediate: bool = False, indent: int = 0):
        """
        Args:
            name: Name of the timing category
            device: CUDA device for GPU timing (None for CPU timing)
            print_immediate: If True, print timing immediately after context exits
            indent: Indentation level for nested timings
        """
        self.name = name
        self.device = device
        self.print_immediate = print_immediate
        self.indent = indent
        self.enabled = TIME_DEBUG_ENABLED
        
        # GPU timing
        self.start_event: Optional[torch.cuda.Event] = None
        self.end_event: Optional[torch.cuda.Event] = None
        
        # CPU timing fallback
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None
        
    def __enter__(self):
        if not self.enabled:
            return self
            
        if self.device is not None and torch.cuda.is_available():
            # Use CUDA events for GPU timing
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            # Use CPU timing
            self.start_time = time.time()
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return False
            
        if self.start_event is not None:
            # GPU timing
            self.end_event.record()
            torch.cuda.synchronize(self.device)
            self.elapsed = self.start_event.elapsed_time(self.end_event) / 1000.0  # Convert ms to seconds
        elif self.start_time is not None:
            # CPU timing
            self.elapsed = time.time() - self.start_time
            
        if self.elapsed is not None:
            # Record in global stats
            _global_stats[self.name].add(self.elapsed)
            
            if self.print_immediate:
                indent_str = "  " * self.indent
                logger.info(f"{indent_str}[TIME_DEBUG] {self.name}: {self.elapsed:.4f}s")
        
        return False
    
    def get_elapsed(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        return self.elapsed


# Global statistics storage
_global_stats: Dict[str, TimingStats] = defaultdict(TimingStats)
_stats_window_size = 10  # Number of steps to aggregate stats


def reset_stats():
    """Reset all timing statistics."""
    _global_stats.clear()


def print_stats(step: int, window_size: int = None):
    """Print aggregated timing statistics.
    
    Args:
        step: Current step number
        window_size: Number of recent steps to include (default: _stats_window_size)
    """
    if not TIME_DEBUG_ENABLED:
        return
        
    window = window_size or _stats_window_size
    
    # Only print if we have enough data
    if not _global_stats:
        return
        
    logger.info(f"[TIME_DEBUG] Stats (last {window} steps, step {step}):")
    
    for name, stats in sorted(_global_stats.items()):
        if stats.count == 0:
            continue
            
        # Get last N measurements
        recent_times = stats.times[-window:] if len(stats.times) > window else stats.times
        if not recent_times:
            continue
            
        mean_time = sum(recent_times) / len(recent_times)
        min_time = min(recent_times)
        max_time = max(recent_times)
        
        logger.info(f"  - {name}: mean={mean_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s")


def print_step_timings(step: int, timings: Dict[str, float], indent: int = 0):
    """Print timings for a single step.
    
    Args:
        step: Step number
        timings: Dictionary of timing name -> elapsed time
        indent: Base indentation level
    """
    if not TIME_DEBUG_ENABLED:
        return
        
    indent_str = "  " * indent
    logger.info(f"{indent_str}[TIME_DEBUG] Step {step}:")
    
    for name, elapsed in sorted(timings.items()):
        logger.info(f"{indent_str}  - {name}: {elapsed:.4f}s")


@contextmanager
def time_block(name: str, device: Optional[torch.device] = None, 
               print_immediate: bool = False, indent: int = 0):
    """Context manager for timing a code block.
    
    Example:
        with time_block("VAE Encode", device=self.device):
            latent = self.vae.encode(video)
    """
    with TimingContext(name, device, print_immediate, indent) as ctx:
        yield ctx


def get_stats_summary() -> Dict[str, Dict[str, float]]:
    """Get summary of all timing statistics.
    
    Returns:
        Dictionary mapping timing name to stats dict with 'mean', 'min', 'max', 'count'
    """
    summary = {}
    for name, stats in _global_stats.items():
        if stats.count > 0:
            summary[name] = {
                'mean': stats.mean(),
                'min': stats.min(),
                'max': stats.max(),
                'count': stats.count,
            }
    return summary

