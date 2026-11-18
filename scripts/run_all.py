"""
run_all.py - run the whole pipeline from raw CT data to final classification, in order.

Run from the project folder:  .venv/bin/python scripts/run_all.py

Steps (each is a separate script, run as its own process so nothing carries over
between steps except the files each step writes):
    1. step1_prepare_data.py          raw KiTS23 volumes -> 72 slices, 8/4 patient split
    2. step2_compare_enhancement.py   Phase 1 comparisons (PSNR, CNR)
    3. step3_compare_segmentation.py  Phase 2 comparisons (Dice, IoU) + learned priors
    4. step4_train_classifier.py      Phase 3 features, feature selection, k-NN, nested CV
    5. step5_evaluate_test_set.py     final evaluation, failure analysis, test-image export

Runs reproducibly from raw CT data to final classification: every setting comes
from config.py and every
random choice uses config.RANDOM_SEED, so two runs give the same numbers.
Stops at the first step that fails.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
STEPS = [
    "step1_prepare_data.py",
    "step2_compare_enhancement.py",
    "step3_compare_segmentation.py",
    "step4_train_classifier.py",
    "step5_evaluate_test_set.py",
]


def run_step(script_name: str) -> float:
    """Run one step with the same Python interpreter; stop everything if it fails. Returns seconds taken."""
    print(f"\n{'=' * 70}\nRunning {script_name}\n{'=' * 70}", flush=True)
    start = time.perf_counter()
    completed = subprocess.run([sys.executable, str(SCRIPTS_DIR / script_name)], cwd=SCRIPTS_DIR.parent)
    if completed.returncode != 0:
        sys.exit(f"\n{script_name} failed (exit code {completed.returncode}) - stopping.")
    return time.perf_counter() - start


def main() -> None:
    durations = {step: run_step(step) for step in STEPS}
    print(f"\n{'=' * 70}\nAll steps finished.")
    for step, seconds in durations.items():
        print(f"    {step:35s} {seconds / 60:5.1f} min")
    print(f"    {'total':35s} {sum(durations.values()) / 60:5.1f} min")


if __name__ == "__main__":
    main()
