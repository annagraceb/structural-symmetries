#!/usr/bin/env python3
"""Master pipeline: Steps 2-7 (and optionally Step 8).

Usage:
    python3 run_pipeline.py           # Steps 2-7
    python3 run_pipeline.py --step8   # Include Step 8 auxiliary loss training
"""

import sys
import os
import time

def main():
    run_step8 = "--step8" in sys.argv

    start = time.time()

    # Step 2: Collect activations
    print("\n" + "=" * 70)
    print("STEP 2: COLLECT ACTIVATIONS")
    print("=" * 70)
    from collect_activations import main as collect_main
    collect_main()

    # Steps 3-7: Analysis pipeline
    print("\n" + "=" * 70)
    print("STEPS 3-7: ANALYSIS PIPELINE")
    print("=" * 70)
    from analysis import main as analysis_main
    analysis_main()

    elapsed = time.time() - start
    print(f"\nSteps 2-7 completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Step 8: Auxiliary loss training
    if run_step8:
        print("\n" + "=" * 70)
        print("STEP 8: AUXILIARY LOSS TRAINING")
        print("=" * 70)
        from run_auxiliary import main as aux_main
        aux_main()

        total = time.time() - start
        print(f"\nFull pipeline (Steps 2-8) completed in {total:.0f}s ({total/60:.1f} min)")
    else:
        print("\nRun with --step8 to also run auxiliary loss training")


if __name__ == "__main__":
    main()
