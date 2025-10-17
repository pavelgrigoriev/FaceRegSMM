import sys
from unittest.mock import MagicMock

try:
    import mamba_ssm
except ImportError:
    sys.modules["mamba_ssm"] = MagicMock()
