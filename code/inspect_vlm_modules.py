# code/inspect_vlm_modules.py
"""One-time inspection: print the real module tree of the loaded VLM so
LoRA target-module names (Task 3) and vlm_hidden_size (Task 5) are taken
from the actual model, not assumed from reading source code alone."""
import sys

sys.path.insert(0, "/workspace/xr1-m7-submission/vendor/Xiaomi-Robotics-1/xr1")
from mibot.models.VLA.XR1 import xr1

model = xr1()
print("vlm_hidden_size:", model.vlm.config.text_config.hidden_size)
print()
print("=== First language_model decoder layer's module names ===")
first_layer = model.vlm.model.language_model.layers[0]
for name, module in first_layer.named_modules():
    if name:
        print(f"{name}: {type(module).__name__}")
