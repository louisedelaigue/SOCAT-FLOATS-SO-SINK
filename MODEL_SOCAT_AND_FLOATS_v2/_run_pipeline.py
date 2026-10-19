import subprocess
import sys

# ======================================================
# List of scripts to run in order
# ======================================================
scripts = [
    "1A_COMBINE_DFS_FROM_FLOATS_AND_SOCAT.py",
    
    #"2_unweighted_SOCAT_AND_FLOATS_SEARCH_ENSEMBLE_MODELS_nolog.py",
    #"3_unweighted_MONTE_CARLO_TEN_BEST_MODELS.py",

    "2_weighted_SOCAT_AND_FLOATS_SEARCH_ENSEMBLE_MODELS_nolog.py",
    "3_weighted_MONTE_CARLO_TEN_BEST_MODELS.py",
    
    #"5_unweighted_SOCAT_AND_FLOATS_ONLY_PREDICT_FLOATS_INDEPENDENT_DATASET.py",
    #"5_unweighted_SOCAT_AND_FLOATS_ONLY_PREDICT_SOCAT_INDEPENDENT_DATASET.py",

    "5_weighted_SOCAT_AND_FLOATS_ONLY_PREDICT_FLOATS_INDEPENDENT_DATASET.py",
    "5_weighted_SOCAT_AND_FLOATS_ONLY_PREDICT_SOCAT_INDEPENDENT_DATASET.py",
    
]

# ======================================================
# Run sequentially
# ======================================================
for script in scripts:
    print(f"\n========================================")
    print(f"Running: {script}")
    print(f"========================================\n")

    result = subprocess.run([sys.executable, script])

    # Stop pipeline if one fails
    if result.returncode != 0:
        print(f"\nERROR: {script} failed. Stopping pipeline.")
        sys.exit(1)

print("\nAll scripts completed successfully.")