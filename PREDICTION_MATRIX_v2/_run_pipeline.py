import subprocess
import sys

# ======================================================
# List of scripts to run in order
# ======================================================
scripts = [
    # "0_download_ARMOR3D.py",
    
    # "1A_start_1x1_ARMOR3D.py",
    # "1B_match_SLA.py",
    # "1C_match_RRS.py",
    # "1D_match_PAR.py",
    # "1E_match_WIND.py",
    # "1F_match_CO2_ATM.py", 
    # "1G_add_coordinates.py",

    "2_PREDICT_FLOATS_ONLY_MODELS.py",
    "2_PREDICT_FLOATS_SUMMER_ONLY_MODELS.py",
    # "2_PREDICT_SOCAT_AND_FLOATS_UNWEIGHTED_MODEL.py",
    "2_PREDICT_SOCAT_AND_FLOATS_WEIGHTED_MODEL.py",
    "2_PREDICT_SOCAT_ONLY_MODELS.py"
    
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
