import subprocess
import sys

# ======================================================
# List of scripts to run in order
# ======================================================
scripts = [
    # "0_prep_floats.py",
    # "1A_add_floats_to_MLD_df.py",

    # "1B_match_SLA.py",
    # "1C_match_RRS.py",
    # "1D_match_PAR.py",
    # "1E_match_wind.py",
    # "1F_match_atmospheric_CO2.py", 
    # "1G_add_coordinates.py",

    # "2_HDBSCAN_split.py",
    "3_FLOATS_ONLY_SEARCH_ENSEMBLE_MODELS_nolog.py",
    "4_MONTE_CARLO_TEN_BEST_MODELS.py",
    # "5_FLOATS_ONLY_PREDICT_INDEPENDENT_DATASET.py"
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
