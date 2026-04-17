# Wrapper script to run the best gradual benchmark from project root
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Now run the actual script
exec(open(PROJECT_ROOT / "experiments" / "main" / "run_best_gradual_benchmark_impl.py").read())