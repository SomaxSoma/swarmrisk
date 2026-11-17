"""SwarmRisk - predict how a company fares against an autonomous AI-agent swarm attack."""
import os
import warnings

# joblib/loky cannot count physical cores on some Windows setups and prints a noisy warning.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

# On some Windows machines loky prints a long traceback while counting CPU cores; it is harmless.
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

__version__ = "1.0.0"
