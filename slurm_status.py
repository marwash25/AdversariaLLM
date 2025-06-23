#!/usr/bin/env python3
"""
Simple SLURM Job Status Visualizer for Multirun Directories
Shows a nice overview of job completion status with color coding.

Usage:
    python3 slurm_status.py           # Show overview of last 20 runs
    python3 slurm_status.py -n 50     # Show last 50 runs
    python3 slurm_status.py -d        # Show detailed view with individual job info
"""

import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import re

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'    # Success
    RED = '\033[91m'      # Failed
    YELLOW = '\033[93m'   # Running/Unknown
    BLUE = '\033[94m'     # Info
    PURPLE = '\033[95m'   # Header
    CYAN = '\033[96m'     # Timestamp
    WHITE = '\033[97m'    # Default
    BOLD = '\033[1m'      # Bold
    END = '\033[0m'       # Reset

def check_job_status(job_dir):
    """Check the status of a single SLURM job"""
    # Look for log files with the correct naming pattern
    log_files = list(job_dir.glob("*_log.out"))

    if not log_files:
        return "NO_LOG", "No log file found"

    log_out = log_files[0]
    log_err = log_out.with_suffix('.err')

    try:
        # Check the last few lines of the output log
        with open(log_out, 'r') as f:
            lines = f.readlines()

        with open(log_err, 'r') as f:
            err_lines = "\n".join(f.readlines()).lower()

        # Look for completion indicators in the last 10 lines
        last_lines = ''.join(lines[-10:]).lower()
        if any(word in err_lines for word in ["error", "failed", "exception", "traceback"]):
            return "FAILED", "Error detected in logs"
        elif "job completed successfully" in last_lines:
            return "SUCCESS", "Completed successfully"
        elif "exiting after successful completion" in last_lines:
            return "SUCCESS", "Completed successfully"
        elif any(word in last_lines for word in ["error", "failed", "exception", "traceback"]):
            return "FAILED", "Error detected in logs"
        else:
            # Check the last few lines of the output log
            last_lines = ''.join(err_lines[-10:]).lower()
            if any(word in last_lines for word in ["error", "failed", "exception", "traceback"]):
                if "out of memory" in last_lines:
                    return "FAILED", "Out of memory"
                if "cancelled at" in last_lines:
                    return "FAILED", "Cancelled"
                return "FAILED", "Error detected in logs"

        # Check if log file was modified in the last 2 minutes (job is likely still running)
        if time.time() - os.path.getmtime(log_out) < 120:
            return "RUNNING", "Job is currently running"

        return "UNKNOWN", "Status unclear"

    except Exception as e:
        return "ERROR", f"Could not read logs: {str(e)}"

def get_run_timestamp(run_path):
    """Extract timestamp from run path for sorting"""
    try:
        date_str = run_path.parent.name  # YYYY-MM-DD
        time_str = run_path.name         # HH-MM-SS
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H-%M-%S")
    except:
        return datetime.min

def get_multirun_status(max_runs=20, detailed=False):
    """Get status of all multirun jobs"""
    multirun_path = Path("multirun")

    if not multirun_path.exists():
        print(f"{Colors.RED}Error: multirun directory not found{Colors.END}")
        return

    # Find all .submitit directories
    submitit_dirs = list(multirun_path.glob("*/*/.submitit"))

    if not submitit_dirs:
        print(f"{Colors.YELLOW}No SLURM jobs found in multirun directory{Colors.END}")
        return

    # Sort by timestamp (most recent first)
    submitit_dirs.sort(key=lambda x: get_run_timestamp(x.parent), reverse=True)

    print(f"{Colors.PURPLE}{Colors.BOLD}🚀 SLURM Job Status Overview{Colors.END}")
    print(f"{Colors.PURPLE}{'='*80}{Colors.END}")

    total_runs = 0
    successful_runs = 0
    failed_runs = 0
    running_runs = 0

    for submitit_dir in submitit_dirs[:max_runs]:
        run_path = submitit_dir.parent
        run_name = f"{run_path.parent.name}/{run_path.name}"
        timestamp = get_run_timestamp(run_path)

        # Find all job directories
        job_dirs = [d for d in submitit_dir.iterdir() if d.is_dir() and re.match(r'\d+_\d+', d.name)]

        if not job_dirs:
            continue

        total_runs += 1

        # Check status of all jobs in this run
        job_statuses = []
        for job_dir in sorted(job_dirs):
            status, message = check_job_status(job_dir)
            job_statuses.append((job_dir.name, status, message))

        # Determine overall run status
        success_count = sum(1 for _, status, _ in job_statuses if status == "SUCCESS")
        failed_count = sum(1 for _, status, _ in job_statuses if status == "FAILED")
        running_count = sum(1 for _, status, _ in job_statuses if status == "RUNNING")
        unknown_count = sum(1 for _, status, _ in job_statuses if status not in ["SUCCESS", "FAILED", "RUNNING"])
        total_jobs = len(job_statuses)

        if success_count == total_jobs:
            run_status = "SUCCESS"
            status_icon = "✅"
            status_color = Colors.GREEN
            successful_runs += 1
        elif failed_count > 0:
            run_status = "FAILED"
            status_icon = "❌"
            status_color = Colors.RED
            failed_runs += 1
        elif running_count > 0:
            run_status = "RUNNING"
            status_icon = "🏃"
            status_color = Colors.YELLOW
            running_runs += 1
        else:
            run_status = "PARTIAL"
            status_icon = "⚠️"
            status_color = Colors.YELLOW

        # Print run summary
        time_str = timestamp.strftime("%m-%d %H:%M") if timestamp != datetime.min else "unknown"
        print(f"{status_color}{status_icon} {run_name}{Colors.END} "
              f"{Colors.CYAN}[{time_str}]{Colors.END} "
              f"({total_jobs} jobs) "
              f"{Colors.GREEN}{success_count}✓{Colors.END} "
              f"{Colors.RED}{failed_count}✗{Colors.END} "
              f"{Colors.YELLOW}{running_count}🏃{Colors.END} "
              f"{Colors.YELLOW}{unknown_count}?{Colors.END}")

        # Show detailed job info if requested
        if detailed and (failed_count > 0 or running_count > 0 or unknown_count > 0):
            for job_name, status, message in job_statuses:
                if status != "SUCCESS":
                    color = Colors.RED if status == "FAILED" else Colors.YELLOW
                    print(f"    {color}└─ {job_name}: {message}{Colors.END}")

    # Print summary
    print(f"\n{Colors.PURPLE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}📊 Summary:{Colors.END}")
    print(f"  Total runs: {total_runs}")
    print(f"  {Colors.GREEN}✅ Successful: {successful_runs}{Colors.END}")
    print(f"  {Colors.RED}❌ Failed: {failed_runs}{Colors.END}")
    print(f"  {Colors.YELLOW}🏃 Running: {running_runs}{Colors.END}")
    print(f"  {Colors.YELLOW}⚠️  Partial/Other: {total_runs - successful_runs - failed_runs - running_runs}{Colors.END}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SLURM Job Status Visualizer")
    parser.add_argument("-n", "--num-runs", type=int, default=20,
                       help="Number of recent runs to show (default: 20)")
    parser.add_argument("-d", "--detailed", action="store_true",
                       help="Show detailed information for failed/unknown jobs")

    args = parser.parse_args()

    try:
        get_multirun_status(max_runs=args.num_runs, detailed=args.detailed)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.END}")

if __name__ == "__main__":
    main()