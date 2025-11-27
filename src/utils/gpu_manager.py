"""
GPU Manager for O-RAG Benchmark

Manages GPU resources for multi-GPU execution on Lambda Cloud.
"""

import os
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import subprocess
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    memory_used_mb: int
    utilization_percent: float
    temperature_c: float = 0.0

    @property
    def memory_utilization(self) -> float:
        if self.memory_total_mb > 0:
            return self.memory_used_mb / self.memory_total_mb
        return 0.0

    @property
    def is_available(self) -> bool:
        """GPU is available if < 90% memory used."""
        return self.memory_utilization < 0.9


@dataclass
class GPUCluster:
    """Information about available GPU cluster."""
    gpus: List[GPUInfo]
    hostname: str = ""
    cuda_version: str = ""

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)

    @property
    def available_gpus(self) -> List[GPUInfo]:
        return [g for g in self.gpus if g.is_available]

    @property
    def total_memory_mb(self) -> int:
        return sum(g.memory_total_mb for g in self.gpus)

    @property
    def free_memory_mb(self) -> int:
        return sum(g.memory_free_mb for g in self.gpus)


class GPUManager:
    """Manage GPU resources for benchmark execution."""

    def __init__(self, device_ids: Optional[List[int]] = None):
        """Initialize GPU manager.

        Args:
            device_ids: Specific GPU IDs to use (None = auto-detect all)
        """
        self.device_ids = device_ids
        self._cluster_info: Optional[GPUCluster] = None

    def detect_gpus(self) -> GPUCluster:
        """Detect available GPUs.

        Returns:
            GPUCluster with GPU information
        """
        gpus = []

        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if self.device_ids is not None and i not in self.device_ids:
                    continue

                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory // (1024 * 1024)

                # Get current memory usage
                torch.cuda.set_device(i)
                memory_free = torch.cuda.memory_reserved(i) // (1024 * 1024)
                memory_used = torch.cuda.memory_allocated(i) // (1024 * 1024)

                gpus.append(GPUInfo(
                    index=i,
                    name=props.name,
                    memory_total_mb=memory_total,
                    memory_free_mb=memory_total - memory_used,
                    memory_used_mb=memory_used,
                    utilization_percent=0.0,  # Would need nvidia-smi
                ))

            cuda_version = torch.version.cuda or "unknown"
        else:
            cuda_version = "N/A"

        hostname = os.uname().nodename if hasattr(os, 'uname') else "unknown"

        self._cluster_info = GPUCluster(
            gpus=gpus,
            hostname=hostname,
            cuda_version=cuda_version,
        )

        return self._cluster_info

    def get_nvidia_smi_info(self) -> Optional[List[Dict]]:
        """Get detailed GPU info from nvidia-smi.

        Returns:
            List of GPU info dicts or None if nvidia-smi unavailable
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return None

            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_free_mb": int(parts[3]),
                        "memory_used_mb": int(parts[4]),
                        "utilization_percent": float(parts[5]),
                        "temperature_c": float(parts[6]),
                    })

            return gpus

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def refresh_gpu_info(self) -> GPUCluster:
        """Refresh GPU information.

        Returns:
            Updated GPUCluster
        """
        nvidia_info = self.get_nvidia_smi_info()

        if nvidia_info:
            gpus = []
            for info in nvidia_info:
                if self.device_ids is not None and info["index"] not in self.device_ids:
                    continue
                gpus.append(GPUInfo(**info))

            hostname = os.uname().nodename if hasattr(os, 'uname') else "unknown"
            cuda_version = "unknown"
            if TORCH_AVAILABLE:
                cuda_version = torch.version.cuda or "unknown"

            self._cluster_info = GPUCluster(
                gpus=gpus,
                hostname=hostname,
                cuda_version=cuda_version,
            )
        else:
            self._cluster_info = self.detect_gpus()

        return self._cluster_info

    @property
    def cluster(self) -> GPUCluster:
        """Get cluster info, detecting if needed."""
        if self._cluster_info is None:
            self.detect_gpus()
        return self._cluster_info

    def get_best_device(self) -> str:
        """Get best available device.

        Returns:
            Device string ('cuda:0', 'cuda:1', or 'cpu')
        """
        cluster = self.cluster

        if not cluster.available_gpus:
            return "cpu"

        # Pick GPU with most free memory
        best = max(cluster.available_gpus, key=lambda g: g.memory_free_mb)
        return f"cuda:{best.index}"

    def get_devices_for_parallel(self, n_workers: int = None) -> List[str]:
        """Get device list for parallel execution.

        Args:
            n_workers: Number of workers (None = use all available)

        Returns:
            List of device strings
        """
        cluster = self.cluster

        if not cluster.available_gpus:
            return ["cpu"]

        available = sorted(cluster.available_gpus, key=lambda g: g.memory_free_mb, reverse=True)

        if n_workers is None:
            n_workers = len(available)

        return [f"cuda:{g.index}" for g in available[:n_workers]]

    def clear_gpu_memory(self, device_id: Optional[int] = None) -> None:
        """Clear GPU memory cache.

        Args:
            device_id: Specific GPU to clear (None = all)
        """
        if not TORCH_AVAILABLE:
            return

        if device_id is not None:
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
        else:
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()

    def wait_for_memory(
        self,
        required_mb: int,
        timeout_seconds: int = 300,
        poll_interval: int = 10,
    ) -> bool:
        """Wait for sufficient GPU memory to become available.

        Args:
            required_mb: Required free memory in MB
            timeout_seconds: Maximum wait time
            poll_interval: Seconds between checks

        Returns:
            True if memory became available, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            self.refresh_gpu_info()

            for gpu in self.cluster.gpus:
                if gpu.memory_free_mb >= required_mb:
                    return True

            self.clear_gpu_memory()
            time.sleep(poll_interval)

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Export cluster info as dict."""
        cluster = self.cluster
        return {
            "hostname": cluster.hostname,
            "cuda_version": cluster.cuda_version,
            "num_gpus": cluster.num_gpus,
            "total_memory_mb": cluster.total_memory_mb,
            "free_memory_mb": cluster.free_memory_mb,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_total_mb": g.memory_total_mb,
                    "memory_free_mb": g.memory_free_mb,
                    "utilization_percent": g.utilization_percent,
                    "is_available": g.is_available,
                }
                for g in cluster.gpus
            ],
        }


def print_gpu_status():
    """Print current GPU status."""
    manager = GPUManager()
    cluster = manager.refresh_gpu_info()

    print(f"\nGPU Status ({cluster.hostname})")
    print(f"CUDA Version: {cluster.cuda_version}")
    print(f"Total GPUs: {cluster.num_gpus}")
    print(f"Total Memory: {cluster.total_memory_mb / 1024:.1f} GB")
    print(f"Free Memory: {cluster.free_memory_mb / 1024:.1f} GB")
    print()

    for gpu in cluster.gpus:
        status = "✓" if gpu.is_available else "✗"
        print(f"  [{status}] GPU {gpu.index}: {gpu.name}")
        print(f"      Memory: {gpu.memory_used_mb}/{gpu.memory_total_mb} MB "
              f"({gpu.memory_utilization*100:.1f}% used)")
        if gpu.temperature_c > 0:
            print(f"      Temp: {gpu.temperature_c}°C")


if __name__ == "__main__":
    print_gpu_status()
