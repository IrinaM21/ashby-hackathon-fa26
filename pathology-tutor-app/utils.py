import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from timm.models import create_model
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

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

    CONCH_REPO_ID = "MahmoodLab/conch"
    CONCH_FILENAME = "pytorch_model.bin"
    CONCH_MODEL_CFG = "conch_ViT-B-16"
    CONCH_LOCAL_PATH = Path(__file__).parent / "weights" / "conch.pt"
    # The public release has no caption decoder weights (removed by MahmoodLab), so CONCH builds the
    # decoder with random weights. This file pins one random init so it's the same on every load,
    # and is where trained decoder weights would go.
    CONCH_DECODER_PATH = Path(__file__).parent / "weights" / "conch_text_decoder.pt"

    @staticmethod
    def load_conch(device=None, weights_path=CONCH_LOCAL_PATH, decoder_path=CONCH_DECODER_PATH):
        """Load the pretrained CONCH model.

        Uses weights_path (written by save_conch_weights) if it exists, so no Hugging Face call;
        otherwise downloads from https://huggingface.co/MahmoodLab/conch. Pass weights_path=None to force the download.
        Loads decoder_path into model.text_decoder if it exists; otherwise the decoder stays randomly initialized.
        Returns (model, preprocess), where preprocess turns a PIL image into the model's input tensor.
        """
        device = torch.device(device) if device is not None else Utils.get_device()
        if weights_path is not None and Path(weights_path).exists():
            ckpt_path = str(weights_path)
        else:
            ckpt_path = hf_hub_download(Utils.CONCH_REPO_ID, filename=Utils.CONCH_FILENAME, token=Utils.get_hf_token())

        model, preprocess = create_model_from_pretrained(Utils.CONCH_MODEL_CFG, checkpoint_path=ckpt_path, device=device)
        if decoder_path is not None and Path(decoder_path).exists():
            model.text_decoder.load_state_dict(torch.load(decoder_path, map_location=device), strict=True)
        model.eval()
        return model, preprocess

    @staticmethod
    def save_conch_weights(path=CONCH_LOCAL_PATH, decoder_path=CONCH_DECODER_PATH, dtype=torch.float32, seed=0):
        """Download CONCH from Hugging Face once and save it locally.

        path gets the released weights (vision + text encoders), identical to the Hugging Face checkpoint.
        decoder_path gets the caption decoder, a seeded random init since no trained weights were released.
        fp32 by default, matching the Hugging Face checkpoint; dtype=torch.float16 halves the file sizes.
        """
        path, decoder_path = Path(path), Path(decoder_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.random.fork_rng():   # seed the decoder init without changing the caller's RNG state
            torch.manual_seed(seed)
            model, _ = Utils.load_conch(device="cpu", weights_path=None, decoder_path=None)

        def cast(sd):
            return {k: v.to(dtype) if v.is_floating_point() else v for k, v in sd.items()}

        state_dict = model.state_dict()
        torch.save(cast({k: v for k, v in state_dict.items() if not k.startswith("text_decoder.")}), path)
        torch.save(cast(model.text_decoder.state_dict()), decoder_path)
        return path, decoder_path

    @staticmethod
    def _generated_token_probs(model_output):
        """Return (generated_ids, token_probs), each (batch, steps), for every generated step."""
        logits = torch.stack(model_output.logits, dim=1)  # (batch, steps, vocab)
        generated_ids = model_output.sequences[:, -logits.shape[1]:]  # (batch, steps)

        probs = torch.softmax(logits.float(), dim=-1)
        token_probs = probs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)  # (batch, steps)
        return generated_ids, token_probs

    @staticmethod
    def get_token_probs(model_output, batch_index=0):
        """Map each generated token id to the probability the model assigned it.

        model_output: result of model.generate(..., return_dict_in_generate=True, output_logits=True),
        with .sequences (token ids) and .logits (one (batch, vocab) tensor per generated step).
        Repeated token ids share a key, so only the last occurrence's probability is kept.
        """
        generated_ids, token_probs = Utils._generated_token_probs(model_output)
        return {
            token_id.item(): prob.item()
            for token_id, prob in zip(generated_ids[batch_index], token_probs[batch_index])
        }

    @staticmethod
    def get_confidence(model_output, batch_index=0):
        """Aggregate generated token probs into one confidence percent (0-100).

        Uses the geometric mean, exp(mean log p), over every generated step, so the
        score doesn't shrink just because the output is longer. Takes the same model_output as get_token_probs.
        """
        _, token_probs = Utils._generated_token_probs(model_output)
        log_probs = token_probs[batch_index].clamp_min(1e-12).log()
        return log_probs.mean().exp().item() * 100

    @staticmethod
    def get_similarity_percent(a, b):
        """Percent similarity (0-100) between a single pair of embeddings (each (d,) or (1, d)).

        Cosine similarity with negatives clipped to 0. Unlike get_match_percents, this is an absolute score.
        """
        cos = F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
        return max(cos, 0.0) * 100

    @staticmethod
    def get_match_percents(model, a, b):
        """Turn image-text cosine similarities into percent match across the text candidates.

        a: (n_images, d), b: (n_texts, d), image or text embeddings with out_norm=True.
        Returns (n_images, n_texts); each row sums to 100. Percents are relative to the candidates given.
        Scales by model.logit_scale as in MUSK's demo.ipynb.
        """
        cos = a.float() @ b.float().T
        return (model.logit_scale.float() * cos).softmax(dim=-1) * 100

    # ---------------------------------------------------------------------------
    # Expert-answer heatmaps (CONCH Grad-CAM), precomputed into cases.json.
    # The app should only draw them once the expert answer is revealed.
    # Re-run make_heatmaps after tools/import_tcga.py, which rewrites cases without the heatmap field.
    # ---------------------------------------------------------------------------

    CASES_PATH = Path(__file__).parent / "data" / "cases.json"
    HEATMAP_METHOD = "gradcam/conch_ViT-B-16/last-block/contrast-normal-tissue"
    # TCGA organ names -> how they read in "normal ... tissue"; other organs fall back to organ.lower().
    _ORGAN_TISSUE = {
        "Breast": "breast",
        "Bronchus and lung": "lung",
        "Colon": "colon",
        "Prostate gland": "prostate",
    }

    @staticmethod
    @torch.no_grad()
    def _conch_text_embedding(model, text):
        device = next(model.parameters()).device
        return model.encode_text(tokenize(texts=[text], tokenizer=get_tokenizer()).to(device), normalize=True)

    @staticmethod
    def _heatmap_direction(model, text, contrast_text):
        """Text-side direction whose dot product with the image embedding is cos(image, text) - cos(image, contrast_text)."""
        return Utils._conch_text_embedding(model, text) - Utils._conch_text_embedding(model, contrast_text)

    @staticmethod
    def _conch_crop_box(model, image_size):
        """Where CONCH's center crop sits on the displayed image, in its pixels: [left, top, right, bottom]."""
        w, h = image_size
        side = model.visual.image_size[0]
        left, top = int(round((w - side) / 2)), int(round((h - side) / 2))   # same rounding as CenterCrop
        return [left, top, left + side, top + side]

    @staticmethod
    def gradcam_heatmap(model, preprocess, image_path, text, contrast_text):
        """Grad-CAM of cos(image, text) - cos(image, contrast_text) on CONCH's last ViT block.

        Highlights regions that make CONCH match the tile to text more than to contrast_text.
        Returns {"grid": side x side list in 0-1, "crop_box": where the grid sits on the displayed image,
        "similarity": cos(image, text), "contrast_similarity": cos(image, contrast_text)}.
        Only the center crop is scored, as in confidence.score_findings.
        """
        device = next(model.parameters()).device
        image = Image.open(image_path).convert("RGB")
        img = preprocess(image).unsqueeze(0).to(device).requires_grad_(True)   # builds the graph
        te = Utils._conch_text_embedding(model, text)
        tc = Utils._conch_text_embedding(model, contrast_text)

        acts = {}
        def keep_tokens(_module, _inputs, out):
            out.retain_grad()
            acts["tokens"] = out
        handle = model.visual.trunk.blocks[-1].register_forward_hook(keep_tokens)
        try:
            with torch.enable_grad():
                ie = model.encode_image(img, proj_contrast=True, normalize=True)
                sim, contrast_sim = (ie @ te.T).squeeze(), (ie @ tc.T).squeeze()
                (sim - contrast_sim).backward()
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)   # backward also filled weight grads; drop them

        n_prefix = model.visual.trunk.num_prefix_tokens
        tokens = acts["tokens"][0, n_prefix:]          # (n_patches, channels)
        grads = acts["tokens"].grad[0, n_prefix:]
        cam = (tokens * grads.mean(0)).sum(-1).relu().detach()
        if cam.max() > 0:
            cam = cam / cam.max()
        side = int(cam.numel() ** 0.5)
        return {
            "grid": cam.reshape(side, side).cpu().round(decimals=3).tolist(),
            "crop_box": Utils._conch_crop_box(model, image.size),
            "similarity": round(sim.item(), 4),
            "contrast_similarity": round(contrast_sim.item(), 4),
        }

    @staticmethod
    @torch.no_grad()
    def heatmap_masking_check(model, preprocess, image_path, text, contrast_text, grid, frac=0.1, n_random=5, seed=0):
        """Sanity check for a heatmap of cos(image, text) - cos(image, contrast_text):
        blanking its top cells should lower that difference more than blanking random cells.

        Blanked patches are set to 0 after normalization, i.e. the dataset mean color.
        """
        device = next(model.parameters()).device
        img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        direction = Utils._heatmap_direction(model, text, contrast_text)
        patch = model.visual.trunk.patch_embed.patch_size[0]
        side = len(grid)

        def sim_without(cells):
            x = img.clone()
            for i in cells:
                r, c = divmod(int(i), side)
                x[..., r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = 0
            return (model.encode_image(x, proj_contrast=True, normalize=True) @ direction.T).item()

        flat = torch.tensor(grid).flatten()
        k = max(1, int(frac * flat.numel()))
        gen = torch.Generator().manual_seed(seed)
        random_sims = [sim_without(torch.randperm(flat.numel(), generator=gen)[:k]) for _ in range(n_random)]
        return {
            "original": sim_without([]),
            "top_blanked": sim_without(flat.topk(k).indices),
            "random_blanked": sum(random_sims) / n_random,
        }

    @staticmethod
    def _expert_text(case):
        ans = case["expert_answer"]
        return f'{ans["diagnosis"]}. {ans["summary"]}'

    @staticmethod
    def _normal_tissue_text(case):
        tissue = Utils._ORGAN_TISSUE.get(case["organ"], case["organ"].lower())
        return f"normal {tissue} tissue"

    @staticmethod
    def _cases_json_dumps(data):
        # Same layout as import_tcga.py, but keep each row of numbers on one line instead of one number per line.
        text = json.dumps(data, indent=2, ensure_ascii=False)
        return re.sub(r"\[\s*([-0-9.eE,\s]+?)\s*\]", lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", text)

    @staticmethod
    def make_heatmaps(cases_path=CASES_PATH, force=False, device=None):
        """Precompute a heatmap of the expert answer vs normal tissue for each case's image and store it in cases.json.

        The map shows what in the tile makes CONCH match the expert answer more than "normal <organ> tissue".
        Skips cases whose image, texts, and method are unchanged unless force=True.
        Run from the app folder: python -c "from utils import Utils; Utils.make_heatmaps()"
        """
        cases_path = Path(cases_path)
        data_dir = cases_path.parent
        data = json.loads(cases_path.read_text())
        model, preprocess = Utils.load_conch(device)

        for case in data["cases"]:
            image_path = data_dir / case["image"]
            text, contrast = Utils._expert_text(case), Utils._normal_tissue_text(case)
            sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
            hm = case.get("heatmap") or {}
            stored = (hm.get("method"), hm.get("text"), hm.get("contrast_text"), hm.get("image_sha256"))
            if not force and stored == (Utils.HEATMAP_METHOD, text, contrast, sha):
                print(f"{case['id']}: up to date")
                continue

            hm = Utils.gradcam_heatmap(model, preprocess, image_path, text, contrast)
            check = Utils.heatmap_masking_check(model, preprocess, image_path, text, contrast, hm["grid"])
            case["heatmap"] = {
                **hm, "text": text, "contrast_text": contrast, "method": Utils.HEATMAP_METHOD, "image_sha256": sha,
            }
            print(
                f"{case['id']}: expert {hm['similarity']:.3f} - normal {hm['contrast_similarity']:.3f}"
                f" = {check['original']:.3f} | top 10% blanked {check['top_blanked']:.3f}"
                f" vs random 10% blanked {check['random_blanked']:.3f}"
            )

        cases_path.write_text(Utils._cases_json_dumps(data))
        print("wrote", cases_path)

    # Jet-style color map anchors: blue (low) -> cyan -> green -> yellow -> red (high).
    _HEATMAP_STOPS = [0.0, 0.25, 0.5, 0.75, 1.0]
    _HEATMAP_COLORS = np.array([[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]], dtype=np.float32)

    @staticmethod
    def render_heatmap(case, data_dir=CASES_PATH.parent, alpha=0.5):
        """Draw a case's stored heatmap (from cases.json) over its image. No model needed.

        Inside crop_box: blue (low) to red (high) at constant opacity alpha. Outside crop_box, which
        CONCH never saw, the image is greyed out, and the box is outlined. Returns a PIL image.
        """
        image = Image.open(Path(data_dir) / case["image"]).convert("RGB")
        hm = case["heatmap"]
        left, top, right, bottom = hm["crop_box"]

        grid = np.asarray(hm["grid"], dtype=np.float32)
        values = np.asarray(Image.fromarray(grid, mode="F").resize((right - left, bottom - top), Image.BILINEAR)).clip(0, 1)
        colors = np.stack(
            [np.interp(values, Utils._HEATMAP_STOPS, Utils._HEATMAP_COLORS[:, ch]) for ch in range(3)], axis=-1
        )
        tile = np.asarray(image.crop((left, top, right, bottom)), dtype=np.float32)
        blended = ((1 - alpha) * tile + alpha * colors).astype(np.uint8)

        out = Image.blend(image, Image.new("RGB", image.size, (128, 128, 128)), 0.6)
        out.paste(Image.fromarray(blended), (left, top))
        ImageDraw.Draw(out).rectangle([left, top, right - 1, bottom - 1], outline=(255, 255, 255), width=3)
        return out


