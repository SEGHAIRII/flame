#!/usr/bin/env python3
"""
Colorful logging utilities for time series training.
Adapted from torchtitan's metrics processor for time series classification.
"""

import time
import torch
from typing import Any, Dict, Optional
from collections import namedtuple


class Color:
    """Color codes for terminal output."""
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    blue = '\033[34m'
    magenta = '\033[35m'
    cyan = '\033[36m'
    orange = '\033[33m'  # Using yellow for orange
    reset = '\033[0m'


class NoColor:
    """No-op color class when colors are disabled."""
    red = ''
    green = ''
    yellow = ''
    blue = ''
    magenta = ''
    cyan = ''
    orange = ''
    reset = ''


# Named tuple for device memory stats
DeviceMemStats = namedtuple(
    "DeviceMemStats",
    [
        "max_reserved_gib",
        "max_reserved_pct",
        "current_reserved_gib",
        "current_reserved_pct",
    ],
)


class DeviceMemoryMonitor:
    """Monitor device memory usage."""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        if device.startswith("cuda") and torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(self.device)
            self.device_capacity = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self.device_name = "CPU"
            self.device_capacity = 0
        
        self.device_capacity_gib = self._to_gib(self.device_capacity)
        self.reset_peak_stats()
    
    def _to_gib(self, memory_in_bytes):
        """Convert bytes to GiB."""
        _gib_in_bytes = 1024 * 1024 * 1024
        return memory_in_bytes / _gib_in_bytes
    
    def _to_pct(self, memory):
        """Convert memory to percentage of total capacity."""
        if self.device_capacity == 0:
            return 0.0
        return 100 * memory / self.device_capacity
    
    def get_memory_stats(self):
        """Get current memory statistics."""
        if not torch.cuda.is_available() or not self.device.type == "cuda":
            return DeviceMemStats(0.0, 0.0, 0.0, 0.0)
        
        current_reserved = torch.cuda.memory_reserved(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)
        
        current_reserved_gib = self._to_gib(current_reserved)
        max_reserved_gib = self._to_gib(max_reserved)
        current_reserved_pct = self._to_pct(current_reserved)
        max_reserved_pct = self._to_pct(max_reserved)
        
        return DeviceMemStats(
            max_reserved_gib,
            max_reserved_pct,
            current_reserved_gib,
            current_reserved_pct,
        )
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)


class TimeSeriesLogger:
    """
    Beautiful logging for time series training similar to the main training script.
    """
    
    def __init__(self, enable_colors: bool = True, device: str = "cuda"):
        self.color = Color() if enable_colors else NoColor()
        self.device_memory_monitor = DeviceMemoryMonitor(device)
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Print device info
        print(f"{self.color.blue}Device: {self.device_memory_monitor.device_name}")
        if self.device_memory_monitor.device_capacity > 0:
            print(f"Memory capacity: {self.device_memory_monitor.device_capacity_gib:.2f}GiB{self.color.reset}")
    
    def log_training_start(self, total_params: int, trainable_params: int, 
                          total_epochs: int, total_steps: int, batch_size: int):
        """Log training start information."""
        print(f"{self.color.red}***** Starting Time Series Training *****{self.color.reset}")
        print(f"{self.color.green}  Total parameters: {total_params:,}")
        print(f"{self.color.green}  Trainable parameters: {trainable_params:,}")
        print(f"{self.color.green}  Total epochs: {total_epochs}")
        print(f"{self.color.green}  Total steps: {total_steps:,}")
        print(f"{self.color.green}  Batch size: {batch_size}{self.color.reset}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        print(f"\n{self.color.cyan}{'='*60}")
        print(f"{self.color.cyan}Epoch {epoch}/{total_epochs}")
        print(f"{'='*60}{self.color.reset}")
    
    def log_training_step(self, epoch: int, step: int, total_steps: int, 
                         loss: float, lr: float, grad_norm: Optional[float] = None):
        """Log training step with beautiful formatting."""
        current_time = time.time()
        time_delta = current_time - self.last_log_time
        
        # Calculate tokens per second (assuming each step processes batch_size samples)
        if time_delta > 0:
            steps_per_sec = 1.0 / time_delta
        else:
            steps_per_sec = 0.0
        
        # Get memory stats
        mem_stats = self.device_memory_monitor.get_memory_stats()
        
        # Calculate elapsed time and ETA
        elapsed_time = current_time - self.start_time
        if step > 0:
            estimated_total_time = elapsed_time * total_steps / step
            eta = estimated_total_time - elapsed_time
            eta_str = self._format_time(eta)
        else:
            eta_str = "unknown"
        
        elapsed_str = self._format_time(elapsed_time)
        
        # Build log message
        log_msg = (
            f"{self.color.red}step: {step:4d}  "
            f"{self.color.green}loss: {loss:7.4f}  "
            f"{self.color.yellow}memory: {mem_stats.current_reserved_gib:5.2f}GiB"
            f"({mem_stats.current_reserved_pct:.2f}%)  "
            f"{self.color.blue}sps: {steps_per_sec:6.2f}  "
            f"{self.color.orange}lr: {lr:.2e}  "
        )
        
        if grad_norm is not None:
            log_msg += f"{self.color.cyan}grad_norm: {grad_norm:6.2f}  "
        
        log_msg += f"{self.color.magenta}[{elapsed_str}<{eta_str}]{self.color.reset}"
        
        print(log_msg)
        self.last_log_time = current_time
    
    def log_validation(self, epoch: int, val_loss: float, val_acc: float, 
                      is_best: bool = False):
        """Log validation results."""
        mem_stats = self.device_memory_monitor.get_memory_stats()
        
        best_marker = f"{self.color.yellow}★ NEW BEST ★{self.color.reset}" if is_best else ""
        
        print(f"{self.color.magenta}validation  "
              f"{self.color.green}loss: {val_loss:7.4f}  "
              f"{self.color.cyan}acc: {val_acc:7.4f}  "
              f"{self.color.yellow}memory: {mem_stats.max_reserved_gib:5.2f}GiB"
              f"({mem_stats.max_reserved_pct:.2f}%)  "
              f"{best_marker}{self.color.reset}")
    
    def log_sequence_metrics(self, epoch: int, metrics: dict, prefix: str = "validation"):
        """Log detailed sequence-level metrics."""
        print(f"\n{self.color.cyan}=== {prefix.title()} Sequence Metrics (Epoch {epoch}) ==={self.color.reset}")
        print(f"{self.color.green}CER: {metrics['cer']:7.4f} ({metrics['cer']*100:6.2f}%)")
        print(f"{self.color.green}Exact Match: {metrics['exact_match']:7.4f} ({metrics['exact_match']*100:6.2f}%)")
        print(f"{self.color.green}Token Accuracy: {metrics['token_accuracy']:7.4f} ({metrics['token_accuracy']*100:6.2f}%)")
        print(f"{self.color.blue}Total Sequences: {metrics['total_sequences']:,}{self.color.reset}")
        print(f"{self.color.cyan}{'='*50}{self.color.reset}")
    
    def log_test_results(self, test_loss: float, test_acc: float):
        """Log final test results."""
        print(f"\n{self.color.red}***** Final Test Results *****{self.color.reset}")
        print(f"{self.color.green}Test Loss: {test_loss:.4f}")
        print(f"{self.color.green}Test Accuracy: {test_acc:.4f}{self.color.reset}")
    
    def log_training_complete(self, total_time: float, best_val_acc: float):
        """Log training completion."""
        print(f"\n{self.color.red}***** Training Complete *****{self.color.reset}")
        print(f"{self.color.green}Total training time: {self._format_time(total_time)}")
        print(f"{self.color.green}Best validation accuracy: {best_val_acc:.4f}{self.color.reset}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds // 60:.0f}m{seconds % 60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"
    
    def log_dataset_info(self, dataset_name: str, n_classes: int, 
                        seq_len: int, input_dim: int, train_size: int, 
                        val_size: int, test_size: int):
        """Log dataset information."""
        print(f"{self.color.blue}Dataset: {dataset_name}")
        print(f"{self.color.blue}  Classes: {n_classes}")
        print(f"{self.color.blue}  Sequence length: {seq_len}")
        print(f"{self.color.blue}  Input dimension: {input_dim}")
        print(f"{self.color.blue}  Train/Val/Test: {train_size}/{val_size}/{test_size}{self.color.reset}")