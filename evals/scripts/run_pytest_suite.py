# Creator: Sulabh Bansod
# Description: Runner script that executes the pytest test suite in tests/ and saves structured results to eval_results/unit_tests_results.json.

import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path


def run_pytest_suite() -> dict:
    """Run pytest suite in tests/ and return structured JSON result."""
    base_dir = Path(__file__).resolve().parents[2]
    tests_dir = base_dir / "tests"
    results_dir = base_dir / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_json = results_dir / "unit_tests_results.json"

    python_bin = sys.executable
    cmd = [python_bin, "-m", "pytest", str(tests_dir), "-v", "--tb=short"]

    start_time = time.perf_counter()
    process = subprocess.run(cmd, cwd=str(base_dir), capture_output=True, text=True)
    duration_sec = round(time.perf_counter() - start_time, 2)

    # Parse stdout for test case results
    passed = 0
    failed = 0
    skipped = 0
    test_cases = []

    lines = process.stdout.splitlines()
    for line in lines:
        if " PASSED" in line:
            passed += 1
            test_name = line.split(" PASSED")[0].strip()
            test_cases.append({"name": test_name, "status": "PASSED", "duration": 0.01})
        elif " FAILED" in line:
            failed += 1
            test_name = line.split(" FAILED")[0].strip()
            test_cases.append({"name": test_name, "status": "FAILED", "duration": 0.01})
        elif " SKIPPED" in line:
            skipped += 1
            test_name = line.split(" SKIPPED")[0].strip()
            test_cases.append({"name": test_name, "status": "SKIPPED", "duration": 0.01})

    summary = {
        "timestamp": datetime.now().isoformat(),
        "duration_sec": duration_sec,
        "total": passed + failed + skipped,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": round(passed / max(passed + failed + skipped, 1) * 100.0, 1),
        "return_code": process.returncode,
        "test_cases": test_cases,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = run_pytest_suite()
    print(f"Pytest suite executed in {res['duration_sec']}s. Passed: {res['passed']}/{res['total']} ({res['pass_rate']}%)")
