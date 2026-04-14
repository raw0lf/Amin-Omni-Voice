import torch
import numpy as np
import soundfile as sf
from omnivoice import OmniVoice

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Check model output shape
try:
    # Minimal attempt to see what generate returns
    # (Might fail due to missing weights, but we just want to see types)
    pass
except Exception as e:
    print(f"Model test error: {e}")
