from pathlib import Path

import torch
import torch.nn as nn
from timm.models import create_model

from utils import Utils

WEIGHTS_DIR = Path(__file__).parent / "weights"


class WeightsUtils:
    FP16_PATH = WEIGHTS_DIR / "musk_fp16.pt"
    INT8_PATH = WEIGHTS_DIR / "musk_int8.pt"

    @staticmethod
    def _set_quantized_engine():
        # The default engine can be "none" (e.g. on Apple silicon); pick one this build supports.
        if torch.backends.quantized.engine == "none":
            engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
            torch.backends.quantized.engine = "fbgemm" if "fbgemm" in engines else engines[0]

    @staticmethod
    def _quantize(model):
        WeightsUtils._set_quantized_engine()
        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    @staticmethod
    def save_musk_weights(path=FP16_PATH, dtype=torch.float16):
        """Download MUSK from Hugging Face once and save its state_dict locally (fp16 by default, ~1.35 GB)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = Utils.load_musk(device="cpu", dtype=torch.float32)
        state_dict = {k: v.to(dtype) if v.is_floating_point() else v for k, v in model.state_dict().items()}
        torch.save(state_dict, path)
        return path

    @staticmethod
    def load_musk_weights(path=FP16_PATH, device=None, dtype=None):
        """Load MUSK from a file written by save_musk_weights, without calling Hugging Face."""
        device = torch.device(device) if device is not None else Utils.get_device()
        if dtype is None:
            dtype = torch.float16 if device.type == "cuda" else torch.float32

        model = create_model(Utils.MODEL_NAME)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(device=device, dtype=dtype)
        model.eval()
        return model

    @staticmethod
    def save_musk_quantized(path=INT8_PATH, source=FP16_PATH):
        """Save an int8 dynamically quantized MUSK (nn.Linear weights only). CPU inference only.

        Builds from the local fp16 file if it exists, otherwise downloads from Hugging Face.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).exists():
            model = WeightsUtils.load_musk_weights(source, device="cpu", dtype=torch.float32)
        else:
            model = Utils.load_musk(device="cpu", dtype=torch.float32)
        torch.save(WeightsUtils._quantize(model).state_dict(), path)
        return path

    @staticmethod
    def load_musk_quantized(path=INT8_PATH):
        """Load the int8 MUSK written by save_musk_quantized. Runs on CPU only."""
        model = create_model(Utils.MODEL_NAME).eval()
        model = WeightsUtils._quantize(model)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
