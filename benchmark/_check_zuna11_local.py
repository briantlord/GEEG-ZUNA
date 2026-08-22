"""Configuration-only validation for the local ZUNA 1.1 Windows run."""

import importlib.metadata as md

import torch
import zuna
import specparam


assert md.version("zuna") == "1.1.3", md.version("zuna")
assert md.version("specparam") == "2.0.0rc7", md.version("specparam")
assert hasattr(zuna, "reconstruct_fif"), zuna.__file__
assert torch.__version__.startswith("2.6.0+cu124"), torch.__version__
assert torch.cuda.is_available(), "CUDA is not available"

print("Python environment: OK")
print("ZUNA:", md.version("zuna"), zuna.__file__)
print("specparam:", md.version("specparam"), specparam.__file__)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
