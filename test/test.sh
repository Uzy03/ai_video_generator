python - <<'PY'
import torch, subprocess
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
subprocess.run(["nvidia-smi"])
PY
