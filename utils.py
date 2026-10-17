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
    
    @staticmethod
    def get_token_probs(model_output, batch_index=0):
        """Map each generated token id to the probability the model assigned it.

        model_output: result of model.generate(..., return_dict_in_generate=True, output_logits=True),
        with .sequences (token ids) and .logits (one (batch, vocab) tensor per generated step).
        Repeated token ids share a key, so only the last occurrence's probability is kept.
        """
        logits = torch.stack(model_output.logits, dim=1)  # (batch, steps, vocab)
        generated_ids = model_output.sequences[:, -logits.shape[1]:]  # (batch, steps)

        probs = torch.softmax(logits.float(), dim=-1)
        token_probs = probs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)  # (batch, steps)

        return {
            token_id.item(): prob.item()
            for token_id, prob in zip(generated_ids[batch_index], token_probs[batch_index])
        }

    @staticmethod
    def get_match_percents(model, image_embeddings, text_embeddings):
        """Turn image-text cosine similarities into percent match across the text candidates.

        image_embeddings: (n_images, d), text_embeddings: (n_texts, d), both from MUSK with out_norm=True.
        Returns (n_images, n_texts); each row sums to 100. Percents are relative to the candidates given.
        Scales by model.logit_scale as in MUSK's demo.ipynb.
        """
        cos = image_embeddings.float() @ text_embeddings.float().T
        return (model.logit_scale.float() * cos).softmax(dim=-1) * 100
    
    

