#!/usr/bin/env python3
"""
Benchmark Monitor

Monitor benchmark progress from local machine via SSH.

Usage:
    python monitor_benchmark.py --host lambda-instance --key ~/.ssh/lambda.pem
    python monitor_benchmark.py --local /path/to/progress.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta


def format_duration(seconds):
    """Format seconds as human-readable duration."""
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"


def fetch_progress_ssh(host, key_file, progress_path):
    """Fetch progress file via SSH."""
    cmd = [
        "ssh", "-i", key_file,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        f"ubuntu@{host}",
        f"cat {progress_path}"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def fetch_logs_ssh(host, key_file, log_path, lines=50):
    """Fetch last N lines of log via SSH."""
    cmd = [
        "ssh", "-i", key_file,
        "-o", "StrictHostKeyChecking=no",
        f"ubuntu@{host}",
        f"tail -n {lines} {log_path}"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout
        return None
    except subprocess.TimeoutExpired:
        return None


def fetch_gpu_status_ssh(host, key_file):
    """Fetch GPU status via SSH."""
    cmd = [
        "ssh", "-i", key_file,
        "-o", "StrictHostKeyChecking=no",
        f"ubuntu@{host}",
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except subprocess.TimeoutExpired:
        return None


def print_status(progress, gpu_status=None, logs=None):
    """Print formatted status."""
    print("\n" + "=" * 60)
    print(f"O-RAG Benchmark Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if progress is None:
        print("\n  ⚠️  Could not fetch progress file")
        return

    status = progress.get("status", "unknown").upper()
    status_emoji = {
        "RUNNING": "🔄",
        "COMPLETED": "✅",
        "FAILED": "❌",
        "PENDING": "⏳",
    }.get(status, "❓")

    print(f"\n{status_emoji} Status: {status}")

    # Progress bar
    percent = progress.get("progress_percent", 0)
    completed = progress.get("completed_tasks", 0)
    total = progress.get("total_tasks", 0)
    bar_width = 30
    filled = int(bar_width * percent / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n  Progress: [{bar}] {percent:.1f}%")
    print(f"  Tasks: {completed}/{total}")

    # Time info
    elapsed = progress.get("elapsed_seconds", 0)
    eta = progress.get("eta_seconds")
    print(f"\n  Elapsed: {format_duration(elapsed)}")
    print(f"  ETA: {format_duration(eta)}")

    if eta:
        completion_time = datetime.now() + timedelta(seconds=eta)
        print(f"  Est. Completion: {completion_time.strftime('%Y-%m-%d %H:%M')}")

    # Current phase
    current_phase = progress.get("current_phase")
    if current_phase:
        print(f"\n  Current Phase: {current_phase}")

    # Errors
    errors = progress.get("errors", [])
    if errors:
        print(f"\n  ⚠️  Errors: {len(errors)}")
        for err in errors[-3:]:  # Show last 3
            print(f"    - {err[:80]}...")

    # Recent results
    phases = progress.get("phases", [])
    for phase in phases:
        if phase.get("status") == "running":
            tasks = phase.get("tasks", [])
            completed_tasks = [t for t in tasks if t.get("status") == "completed"]
            if completed_tasks:
                print(f"\n  Recent Results:")
                for task in completed_tasks[-5:]:  # Last 5
                    result = task.get("result", {})
                    mrr = result.get("mrr@5", 0)
                    print(f"    {task.get('task_name', 'unknown'):30} MRR@5: {mrr:.4f}")

    # GPU status
    if gpu_status:
        print(f"\n  GPU Status:")
        for line in gpu_status.split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    print(f"    GPU {parts[0]}: {parts[2]}/{parts[3]} MB, "
                          f"{parts[4]}% util, {parts[5]}°C")

    # Recent logs
    if logs:
        print(f"\n  Recent Log Output:")
        log_lines = logs.strip().split('\n')[-10:]  # Last 10 lines
        for line in log_lines:
            print(f"    {line}")

    print("\n" + "=" * 60)


def monitor_loop(args):
    """Continuous monitoring loop."""
    while True:
        if args.local:
            # Local file
            try:
                with open(args.local) as f:
                    progress = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                progress = None
            gpu_status = None
            logs = None
        else:
            # Remote via SSH
            progress = fetch_progress_ssh(
                args.host, args.key, args.progress_path
            )
            if args.gpu:
                gpu_status = fetch_gpu_status_ssh(args.host, args.key)
            else:
                gpu_status = None
            if args.logs:
                logs = fetch_logs_ssh(args.host, args.key, args.log_path)
            else:
                logs = None

        print_status(progress, gpu_status, logs)

        # Check if complete
        if progress and progress.get("status") in ["completed", "failed"]:
            print("\nBenchmark finished!")
            break

        if not args.continuous:
            break

        print(f"\nNext check in {args.interval} seconds... (Ctrl+C to stop)")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Monitor O-RAG benchmark progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # SSH options
    parser.add_argument("--host", type=str, help="Lambda instance hostname/IP")
    parser.add_argument("--key", type=str, default="~/.ssh/lambda.pem",
                        help="SSH key file")
    parser.add_argument("--progress-path", type=str,
                        default="/home/ubuntu/o-rag/o-ragreltesting/results/progress.json",
                        help="Path to progress.json on remote")
    parser.add_argument("--log-path", type=str,
                        default="/home/ubuntu/persistent/logs/benchmark.log",
                        help="Path to log file on remote")

    # Local option
    parser.add_argument("--local", type=Path, help="Local progress.json file")

    # Display options
    parser.add_argument("--gpu", action="store_true", help="Show GPU status")
    parser.add_argument("--logs", action="store_true", help="Show recent logs")

    # Monitoring options
    parser.add_argument("--continuous", "-c", action="store_true",
                        help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between checks (default: 60)")

    args = parser.parse_args()

    if not args.local and not args.host:
        print("Error: Must specify either --host or --local")
        parser.print_help()
        sys.exit(1)

    if args.key:
        args.key = str(Path(args.key).expanduser())

    monitor_loop(args)


if __name__ == "__main__":
    main()
