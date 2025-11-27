"""
Progress Tracker for O-RAG Benchmark

Tracks benchmark progress with persistence for monitoring long-running jobs.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import atexit


class TaskStatus(str, Enum):
    """Status of a benchmark task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskProgress:
    """Progress of a single task."""
    task_id: str
    task_name: str
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start(self):
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now().isoformat()

    def complete(self, result: Optional[Dict] = None):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self.result = result
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            self.duration_seconds = (datetime.now() - start).total_seconds()

    def fail(self, error: str):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.error = error
        self.retry_count += 1
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            self.duration_seconds = (datetime.now() - start).total_seconds()

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "TaskProgress":
        d["status"] = TaskStatus(d["status"])
        return cls(**d)


@dataclass
class PhaseProgress:
    """Progress of a benchmark phase."""
    phase_id: str
    phase_name: str
    tasks: List[TaskProgress] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def completed_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)

    @property
    def failed_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)

    @property
    def progress_percent(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    def start(self):
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now().isoformat()

    def complete(self):
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "phase_id": self.phase_id,
            "phase_name": self.phase_name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "progress_percent": self.progress_percent,
            "tasks": [t.to_dict() for t in self.tasks],
        }


@dataclass
class BenchmarkProgress:
    """Overall benchmark progress."""
    benchmark_id: str
    benchmark_name: str
    phases: List[PhaseProgress] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def current_phase(self) -> Optional[PhaseProgress]:
        for phase in self.phases:
            if phase.status == TaskStatus.RUNNING:
                return phase
        return None

    @property
    def total_tasks(self) -> int:
        return sum(p.total_tasks for p in self.phases)

    @property
    def completed_tasks(self) -> int:
        return sum(p.completed_tasks for p in self.phases)

    @property
    def progress_percent(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100

    @property
    def elapsed_time(self) -> float:
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            return (datetime.now() - start).total_seconds()
        return 0.0

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.completed_tasks == 0 or self.elapsed_time == 0:
            return None
        rate = self.completed_tasks / self.elapsed_time
        remaining = self.total_tasks - self.completed_tasks
        return remaining / rate if rate > 0 else None

    def to_dict(self) -> Dict:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "progress_percent": self.progress_percent,
            "elapsed_seconds": self.elapsed_time,
            "eta_seconds": self.eta_seconds,
            "current_phase": self.current_phase.phase_name if self.current_phase else None,
            "config": self.config,
            "metrics": self.metrics,
            "errors": self.errors,
            "phases": [p.to_dict() for p in self.phases],
        }


class ProgressTracker:
    """Track and persist benchmark progress."""

    def __init__(
        self,
        progress_file: Path,
        auto_save_interval: int = 30,
    ):
        """Initialize tracker.

        Args:
            progress_file: Path to progress JSON file
            auto_save_interval: Seconds between auto-saves (0 to disable)
        """
        self.progress_file = Path(progress_file)
        self.auto_save_interval = auto_save_interval
        self.progress: Optional[BenchmarkProgress] = None
        self._lock = threading.Lock()
        self._auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()

        # Register cleanup
        atexit.register(self._cleanup)

    def _cleanup(self):
        """Cleanup on exit."""
        self._stop_auto_save.set()
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self._auto_save_thread.join(timeout=5)
        self.save()

    def _auto_save_loop(self):
        """Background thread for auto-saving."""
        while not self._stop_auto_save.is_set():
            self._stop_auto_save.wait(self.auto_save_interval)
            if not self._stop_auto_save.is_set():
                self.save()

    def create_benchmark(
        self,
        benchmark_id: str,
        benchmark_name: str,
        phases: List[Dict[str, Any]],
        config: Dict[str, Any] = None,
    ) -> BenchmarkProgress:
        """Create a new benchmark.

        Args:
            benchmark_id: Unique identifier
            benchmark_name: Human-readable name
            phases: List of {"phase_id", "phase_name", "tasks": [...]}
            config: Benchmark configuration

        Returns:
            BenchmarkProgress instance
        """
        with self._lock:
            phase_objects = []
            for phase_def in phases:
                tasks = [
                    TaskProgress(
                        task_id=t.get("task_id", f"task_{i}"),
                        task_name=t.get("task_name", f"Task {i}"),
                        metadata=t.get("metadata", {}),
                    )
                    for i, t in enumerate(phase_def.get("tasks", []))
                ]
                phase_objects.append(PhaseProgress(
                    phase_id=phase_def["phase_id"],
                    phase_name=phase_def["phase_name"],
                    tasks=tasks,
                ))

            self.progress = BenchmarkProgress(
                benchmark_id=benchmark_id,
                benchmark_name=benchmark_name,
                phases=phase_objects,
                config=config or {},
            )

            self.save()

            # Start auto-save thread
            if self.auto_save_interval > 0:
                self._stop_auto_save.clear()
                self._auto_save_thread = threading.Thread(
                    target=self._auto_save_loop,
                    daemon=True,
                )
                self._auto_save_thread.start()

            return self.progress

    def start_benchmark(self):
        """Mark benchmark as started."""
        with self._lock:
            if self.progress:
                self.progress.status = TaskStatus.RUNNING
                self.progress.started_at = datetime.now().isoformat()
                self.save()

    def complete_benchmark(self, metrics: Dict[str, Any] = None):
        """Mark benchmark as completed."""
        with self._lock:
            if self.progress:
                self.progress.status = TaskStatus.COMPLETED
                self.progress.completed_at = datetime.now().isoformat()
                if metrics:
                    self.progress.metrics = metrics
                self.save()

    def fail_benchmark(self, error: str):
        """Mark benchmark as failed."""
        with self._lock:
            if self.progress:
                self.progress.status = TaskStatus.FAILED
                self.progress.completed_at = datetime.now().isoformat()
                self.progress.errors.append(error)
                self.save()

    def start_phase(self, phase_id: str):
        """Mark phase as started."""
        with self._lock:
            if self.progress:
                for phase in self.progress.phases:
                    if phase.phase_id == phase_id:
                        phase.start()
                        break
                self.save()

    def complete_phase(self, phase_id: str):
        """Mark phase as completed."""
        with self._lock:
            if self.progress:
                for phase in self.progress.phases:
                    if phase.phase_id == phase_id:
                        phase.complete()
                        break
                self.save()

    def start_task(self, phase_id: str, task_id: str):
        """Mark task as started."""
        with self._lock:
            task = self._find_task(phase_id, task_id)
            if task:
                task.start()
                self.save()

    def complete_task(
        self,
        phase_id: str,
        task_id: str,
        result: Dict[str, Any] = None,
    ):
        """Mark task as completed."""
        with self._lock:
            task = self._find_task(phase_id, task_id)
            if task:
                task.complete(result)
                self.save()

    def fail_task(self, phase_id: str, task_id: str, error: str):
        """Mark task as failed."""
        with self._lock:
            task = self._find_task(phase_id, task_id)
            if task:
                task.fail(error)
                self.save()

    def _find_task(self, phase_id: str, task_id: str) -> Optional[TaskProgress]:
        """Find a task by phase and task ID."""
        if not self.progress:
            return None
        for phase in self.progress.phases:
            if phase.phase_id == phase_id:
                for task in phase.tasks:
                    if task.task_id == task_id:
                        return task
        return None

    def save(self):
        """Save progress to file."""
        if self.progress:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress.to_dict(), f, indent=2)

    def load(self) -> Optional[BenchmarkProgress]:
        """Load progress from file.

        Returns:
            BenchmarkProgress if file exists, None otherwise
        """
        if not self.progress_file.exists():
            return None

        with open(self.progress_file) as f:
            data = json.load(f)

        # Reconstruct objects
        phases = []
        for phase_data in data.get("phases", []):
            tasks = [
                TaskProgress.from_dict(t)
                for t in phase_data.get("tasks", [])
            ]
            phases.append(PhaseProgress(
                phase_id=phase_data["phase_id"],
                phase_name=phase_data["phase_name"],
                tasks=tasks,
                status=TaskStatus(phase_data["status"]),
                started_at=phase_data.get("started_at"),
                completed_at=phase_data.get("completed_at"),
            ))

        self.progress = BenchmarkProgress(
            benchmark_id=data["benchmark_id"],
            benchmark_name=data["benchmark_name"],
            phases=phases,
            status=TaskStatus(data["status"]),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            config=data.get("config", {}),
            metrics=data.get("metrics", {}),
            errors=data.get("errors", []),
        )

        return self.progress

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current status summary.

        Returns:
            Status summary dict
        """
        if not self.progress:
            return {"status": "not_started"}

        return {
            "status": self.progress.status.value,
            "progress_percent": round(self.progress.progress_percent, 1),
            "completed_tasks": self.progress.completed_tasks,
            "total_tasks": self.progress.total_tasks,
            "elapsed_time": format_duration(self.progress.elapsed_time),
            "eta": format_duration(self.progress.eta_seconds) if self.progress.eta_seconds else "unknown",
            "current_phase": self.progress.current_phase.phase_name if self.progress.current_phase else None,
            "errors": len(self.progress.errors),
        }

    def print_status(self):
        """Print current status to console."""
        summary = self.get_status_summary()

        print(f"\n{'='*60}")
        print(f"Benchmark Status: {summary['status'].upper()}")
        print(f"Progress: {summary['completed_tasks']}/{summary['total_tasks']} tasks ({summary['progress_percent']}%)")
        print(f"Elapsed: {summary['elapsed_time']}")
        print(f"ETA: {summary['eta']}")
        if summary['current_phase']:
            print(f"Current Phase: {summary['current_phase']}")
        if summary['errors'] > 0:
            print(f"Errors: {summary['errors']}")
        print(f"{'='*60}\n")


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in human-readable form."""
    if seconds is None:
        return "unknown"

    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def create_phase_definitions(
    strategies: List[str],
    phase_name: str = "Strategy Evaluation",
) -> List[Dict]:
    """Helper to create phase definitions from strategy list.

    Args:
        strategies: List of strategy IDs
        phase_name: Name for the phase

    Returns:
        Phase definition list
    """
    return [{
        "phase_id": "strategy_eval",
        "phase_name": phase_name,
        "tasks": [
            {"task_id": sid, "task_name": f"Evaluate {sid}"}
            for sid in strategies
        ],
    }]


if __name__ == "__main__":
    # Demo usage
    tracker = ProgressTracker(Path("/tmp/test_progress.json"))

    # Create benchmark
    phases = create_phase_definitions(
        ["c1_dense", "c2_bm25", "c7_rrf"],
        "Test Phase"
    )
    tracker.create_benchmark("test_001", "Test Benchmark", phases)

    # Simulate progress
    tracker.start_benchmark()
    tracker.start_phase("strategy_eval")

    for sid in ["c1_dense", "c2_bm25", "c7_rrf"]:
        tracker.start_task("strategy_eval", sid)
        time.sleep(0.5)
        tracker.complete_task("strategy_eval", sid, {"mrr@5": 0.8})
        tracker.print_status()

    tracker.complete_phase("strategy_eval")
    tracker.complete_benchmark({"best_strategy": "c7_rrf"})

    print("\nFinal status:")
    tracker.print_status()
