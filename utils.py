import os

import torch
from huggingface_hub import hf_hub_download
from timm.models import create_model

from musk import utils as musk_utils
from musk import modeling  # noqa: F401  registers musk_large_patch16_384 with timm


class Utils:
    HF_REPO_ID = "xiangjx/musk"
    HF_FILENAME = "model.safetensors"
    MODEL_NAME = "musk_large_patch16_384"

    @staticmethod
    def get_hf_token():
        """Return the Hugging Face token from the HF_TOKEN environment variable, or None if unset."""
        return os.environ.get("HF_TOKEN") or None

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def load_musk(device=None, dtype=None):
        """Load the pretrained MUSK model from https://huggingface.co/xiangjx/musk.

        Defaults to float16 on CUDA and float32 elsewhere.
        """
        device = torch.device(device) if device is not None else Utils.get_device()
        if dtype is None:
            dtype = torch.float16 if device.type == "cuda" else torch.float32

        ckpt_path = hf_hub_download(Utils.HF_REPO_ID, filename=Utils.HF_FILENAME, token=Utils.get_hf_token())

        model = create_model(Utils.MODEL_NAME)
        musk_utils.load_model_and_may_interpolate(ckpt_path, model, "model|module", "")
        model.to(device=device, dtype=dtype)
        model.eval()
        return model
