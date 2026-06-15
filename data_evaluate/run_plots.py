import subprocess
import os
import sys

# Change environment variable so matplotlib uses non-interactive backend
os.environ["MPLBACKEND"] = "Agg"

# Get absolute path of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("--- Step 1: Converting notebook to Python script ---")
try:
    subprocess.run([
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "script",
        "evaluate.ipynb"
    ], check=True)
except Exception as e:
    print(f"Error converting notebook: {e}")
    sys.exit(1)

print("\n--- Step 2: Running evaluate.py to generate 82 charts ---")
try:
    subprocess.run([
        sys.executable, "evaluate.py"
    ], check=True)
except Exception as e:
    print(f"Error running evaluate.py: {e}")
    # Don't delete yet if there's an error so the user can debug
    sys.exit(1)

# Clean up
if os.path.exists("evaluate.py"):
    os.remove("evaluate.py")

print("\n=== SUCCESS: All charts regenerated in data_evaluate/charts/ ===")
