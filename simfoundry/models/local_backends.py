# SPDX-License-Identifier: Apache-2.0
"""Local stand-ins for the remote Gemini endpoints.

Enabled with SIMFOUNDRY_LOCAL_BACKEND=1 (see ``vlm._get_genai_module``). The pipeline keeps calling
``Gemini(...)`` exactly as before; only the client underneath is swapped:

- text/vision models (gemini-*-pro, -flash, ...) -> a local Ollama serving Qwen3-VL
- image models (anything with "image" in the name, e.g. gemini-3-pro-image)
                                                   -> a local ComfyUI running Qwen-Image-Edit-2511

Box/point conventions differ: Gemini answers ``box_2d`` as [ymin, xmin, ymax, xmax] and points as
[y, x] (0-1000), while Qwen3-VL emits [x1, y1, x2, y2] / [x, y] in the same 0-1000 frame and keeps
doing so even when a prompt asks otherwise. We pin Qwen to its native order with a system message
and swap the coordinates back on the way out, so callers see Gemini's convention.
"""

import base64
import contextlib
import io
import json
import os
import re
import time
import uuid
from types import SimpleNamespace

import requests
from PIL import Image

OLLAMA_URL = os.environ.get("SIMFOUNDRY_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("SIMFOUNDRY_OLLAMA_MODEL", "qwen3-vl:8b-instruct")
OLLAMA_KEEP_ALIVE = os.environ.get("SIMFOUNDRY_OLLAMA_KEEP_ALIVE", "60s")
OLLAMA_NUM_CTX = int(os.environ.get("SIMFOUNDRY_OLLAMA_NUM_CTX", "16384"))
OLLAMA_MAX_TOKENS = int(os.environ.get("SIMFOUNDRY_OLLAMA_MAX_TOKENS", "4096"))
# Layers to place on the GPU; unset lets Ollama fit what is free, 0 keeps the VLM entirely on CPU
# so it never competes with SAM3 / ComfyUI for the 12 GB card.
OLLAMA_NUM_GPU = os.environ.get("SIMFOUNDRY_OLLAMA_NUM_GPU")
# Whether a thinking model reasons before answering: 0 sends think=false, 1 think=true; unset sends
# nothing and the model keeps its default. Qwen3.5 thinks by default and needs 0: its reasoning counts
# against OLLAMA_MAX_TOKENS, and while thinking it writes boxes in the [y, x] order a prompt asks for
# rather than the order SYSTEM_PROMPT pins, so to_gemini_coordinates would swap them the wrong way.
OLLAMA_THINK = os.environ.get("SIMFOUNDRY_OLLAMA_THINK")
# Long side of images sent to the VLM. Coordinates are normalized, so downscaling only costs detail;
# it bounds the visual-token count (and CPU time when the GPU is busy).
VLM_MAX_SIDE = int(os.environ.get("SIMFOUNDRY_LOCAL_VLM_MAX_SIDE", "1280"))

COMFY_URL = os.environ.get("SIMFOUNDRY_COMFY_URL", "http://127.0.0.1:8189")
COMFY_UNET = os.environ.get("SIMFOUNDRY_COMFY_UNET", "qwen-image-edit-2511-Q3_K_M.gguf")
COMFY_LORA = os.environ.get("SIMFOUNDRY_COMFY_LORA", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
COMFY_CLIP = os.environ.get("SIMFOUNDRY_COMFY_CLIP", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
COMFY_VAE = os.environ.get("SIMFOUNDRY_COMFY_VAE", "qwen_image_vae.safetensors")
COMFY_STEPS = int(os.environ.get("SIMFOUNDRY_COMFY_STEPS", "4"))
COMFY_MEGAPIXELS = float(os.environ.get("SIMFOUNDRY_COMFY_MEGAPIXELS", "1.0"))
COMFY_TIMEOUT_S = float(os.environ.get("SIMFOUNDRY_COMFY_TIMEOUT_S", "1800"))
# Which ComfyUI image model serves the Gemini image calls: "edit2511" (Qwen-Image-Edit-2511 + 4-step
# Lightning LoRA, the default used for the reported runs) or "qwen21" (Qwen-Image-2.1-Turbo, a unified
# generation/editing model that natively takes painted annotations and produces RGBA).
IMAGE_BACKEND = os.environ.get("SIMFOUNDRY_IMAGE_BACKEND", "edit2511")
Q21_UNET = os.environ.get("SIMFOUNDRY_Q21_UNET", "qwen-image-2.1-turbo-Q8_0.gguf")
Q21_CLIP = os.environ.get("SIMFOUNDRY_Q21_CLIP", "qwen3vl_8b_int8_convrot.safetensors")
Q21_VAE = os.environ.get("SIMFOUNDRY_Q21_VAE", "qwen_image_2.1_vae_bf16.safetensors")
# The Turbo checkpoint is distilled for 8 steps at CFG 1 (model card).
Q21_STEPS = int(os.environ.get("SIMFOUNDRY_Q21_STEPS", "8"))
Q21_RESOLUTION = int(os.environ.get("SIMFOUNDRY_Q21_RESOLUTION", "1024"))
# Unload ComfyUI's models after every job so pipeline stages get the GPU back.
COMFY_FREE_AFTER = os.environ.get("SIMFOUNDRY_COMFY_FREE_AFTER", "1") == "1"

SYSTEM_PROMPT = (
    "You are a careful vision-language assistant inside an automated robotics pipeline. "
    "Follow the requested output format exactly; when JSON is requested, output only valid JSON. "
    "Coordinate rule: whenever you output a bounding box, ALWAYS write it as [x_min, y_min, x_max, y_max] "
    "and whenever you output a point, ALWAYS write it as [x, y], both normalized to 0-1000, even if the "
    "request describes a different order; the order is converted automatically afterwards."
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def is_enabled():
    return os.environ.get("SIMFOUNDRY_LOCAL_BACKEND", "").strip().lower() in {"1", "true", "yes", "on"}


def sam3_checkpoint_kwargs():
    """kwargs for ``build_sam3_image_model`` pointing at a local checkpoint.

    facebook/sam3 is gated on Hugging Face; SIMFOUNDRY_SAM3_CHECKPOINT names a local ``.pt`` whose
    ``model`` dict uses the upstream ``detector.*`` / ``tracker.*`` key layout instead.
    """
    path = os.environ.get("SIMFOUNDRY_SAM3_CHECKPOINT")
    if not path:
        return {}
    return {"checkpoint_path": path, "load_from_HF": False}


def _is_image_model(model):
    return "image" in model.lower()


# ----------------------------------------------------------------------------------------------
# Request decoding
# ----------------------------------------------------------------------------------------------

def _image_bytes(data):
    """Raw image bytes from a Part's inline data.

    ``Gemini.__call__`` hands ``Part.from_bytes`` a base64 *string*; depending on whether the real
    google-genai types or the fallback types are in use, it arrives here as that string, its UTF-8
    bytes, or already-decoded bytes.
    """
    if isinstance(data, str):
        data = data.encode("ascii")
    data = bytes(data)
    if data.startswith(_PNG_MAGIC) or data[:3] == b"\xff\xd8\xff" or data[:4] == b"RIFF":
        return data
    return base64.b64decode(data)


def _flatten_contents(contents):
    """(text, [image bytes]) from whatever ``contents`` shape the caller passed."""
    texts, images = [], []

    def visit(obj):
        if obj is None:
            return
        if isinstance(obj, str):
            texts.append(obj)
            return
        if isinstance(obj, (list, tuple)):
            for o in obj:
                visit(o)
            return
        parts = getattr(obj, "parts", None)
        if parts is not None:
            visit(list(parts))
            return
        inline = getattr(obj, "inline_data", None)
        if inline is not None and getattr(inline, "data", None) is not None:
            images.append(_image_bytes(inline.data))
            return
        text = getattr(obj, "text", None)
        if text is not None:
            texts.append(text)

    visit(contents)
    return "\n\n".join(texts), images


def _config_value(config, name, default=None):
    value = getattr(config, name, None) if config is not None else None
    return default if value is None else value


def _resize_for_vlm(img_bytes):
    image = _flatten_on_white(Image.open(io.BytesIO(img_bytes)))
    scale = VLM_MAX_SIDE / max(image.size)
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------------------------
# Coordinate conversion (Qwen xy-order -> Gemini yx-order)
# ----------------------------------------------------------------------------------------------

_NUM = r"-?\d+(?:\.\d+)?"
_BOX_KEYS = r"box_2d|bbox|bounding_box|box"
_POINT_KEYS = r"point|point_2d"
_BOX_RE = re.compile(rf'("(?:{_BOX_KEYS})"\s*:\s*)\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]')
_POINT_RE = re.compile(rf'("(?:{_POINT_KEYS})"\s*:\s*)\[\s*({_NUM})\s*,\s*({_NUM})\s*\]')


def _wants_yx_order(prompt):
    p = prompt.lower().replace(" ", "")
    return ("box_2d" in p) or ("ymin,xmin" in p) or ("[y,x]" in p) or ("y,x]" in p and "point" in p)


def to_gemini_coordinates(text):
    """Swap Qwen's [x1,y1,x2,y2]/[x,y] values under box/point keys into Gemini's [y,x] order."""
    text = _BOX_RE.sub(lambda m: f"{m.group(1)}[{m.group(3)}, {m.group(2)}, {m.group(5)}, {m.group(4)}]", text)
    text = _POINT_RE.sub(lambda m: f"{m.group(1)}[{m.group(3)}, {m.group(2)}]", text)
    return text


# ----------------------------------------------------------------------------------------------
# Response objects shaped like google.genai GenerateContentResponse chunks
# ----------------------------------------------------------------------------------------------

def _response(text=None, images=(), finish_reason="STOP"):
    parts = []
    if text:
        parts.append(SimpleNamespace(text=text, inline_data=None, thought=None))
    for img in images:
        parts.append(SimpleNamespace(text=None, thought=None, inline_data=SimpleNamespace(data=img, mime_type="image/png")))
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts, role="model"), finish_reason=finish_reason)
    return SimpleNamespace(text=text, candidates=[candidate], prompt_feedback=None)


# ----------------------------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------------------------

def ollama_chat(prompt, images, temperature=0.0, seed=0, max_tokens=None, verbose=False):
    body = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": [_resize_for_vlm(b) for b in images]},
        ],
        "options": {
            "temperature": float(temperature or 0.0),
            "seed": int(seed or 0),
            "num_predict": int(min(max_tokens or OLLAMA_MAX_TOKENS, OLLAMA_MAX_TOKENS)),
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    if OLLAMA_NUM_GPU not in (None, ""):
        body["options"]["num_gpu"] = int(OLLAMA_NUM_GPU)
    if OLLAMA_THINK not in (None, ""):
        body["think"] = OLLAMA_THINK.strip().lower() in {"1", "true", "yes", "on"}
    try:
        for attempt in range(2):
            t0 = time.time()
            resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=3600)
            resp.raise_for_status()
            out = resp.json()
            text = out["message"]["content"]
            if verbose:
                print(f"[local-vlm] {OLLAMA_MODEL}: {len(images)} image(s), {out.get('prompt_eval_count')} prompt / "
                      f"{out.get('eval_count')} gen tokens in {time.time() - t0:.1f}s", flush=True)
            if out.get("done_reason") != "length":
                return text, "STOP"
            # Greedy decoding on long chain-of-thought prompts sometimes falls into a repetition loop
            # and runs to the token cap; one sampled retry usually escapes it.
            body["options"]["temperature"] = max(body["options"]["temperature"], 0.6)
            body["options"]["seed"] += 7919
        return text, "MAX_TOKENS"
    finally:
        if OLLAMA_KEEP_ALIVE in ("0", "0s"):
            _wait_until(lambda: not requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models"),
                        timeout_s=60)


def _wait_until(predicate, timeout_s, interval_s=0.5):
    """Poll ``predicate`` until it is true or ``timeout_s`` passes; returns whether it became true."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except requests.RequestException:
            pass
        time.sleep(interval_s)
    return False


def _flatten_on_white(image):
    """RGB image with any transparency composited onto white.

    LoadImage drops alpha and keeps the colour underneath it, and stage 6 / B1 cutouts pad with
    (0,0,0,0) — sent as-is the model would see a black frame around the object.
    """
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return image.convert("RGB")


def _comfy_upload(img_bytes):
    name = f"simfoundry_{uuid.uuid4().hex[:12]}.png"
    image = _flatten_on_white(Image.open(io.BytesIO(img_bytes)))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    resp = requests.post(
        f"{COMFY_URL}/upload/image",
        files={"image": (name, buf.getvalue(), "image/png")},
        data={"type": "input", "subfolder": "simfoundry_local", "overwrite": "true"},
        timeout=120,
    )
    resp.raise_for_status()
    info = resp.json()
    return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]


def _qwen_edit_graph(prompt, image_names, seed, width=1024, height=1024):
    g = {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": COMFY_UNET}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": COMFY_LORA, "strength_model": 1.0}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lora", 0], "shift": 3.1}},
        "cfgnorm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": COMFY_CLIP, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": COMFY_VAE}},
    }
    pos = {"clip": ["clip", 0], "prompt": prompt, "vae": ["vae", 0]}
    neg = {"clip": ["clip", 0], "prompt": "", "vae": ["vae", 0]}
    for i, name in enumerate(image_names[:3]):
        g[f"load{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[f"scale{i}"] = {"class_type": "ImageScaleToTotalPixels",
                          "inputs": {"image": [f"load{i}", 0], "upscale_method": "lanczos",
                                     "megapixels": COMFY_MEGAPIXELS, "resolution_steps": 16}}
        pos[f"image{i + 1}"] = [f"scale{i}", 0]
        neg[f"image{i + 1}"] = [f"scale{i}", 0]
    g["pos"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": pos}
    g["neg"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": neg}
    g["pos_ref"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                    "inputs": {"conditioning": ["pos", 0], "reference_latents_method": "index_timestep_zero"}}
    g["neg_ref"] = {"class_type": "FluxKontextMultiReferenceLatentMethod",
                    "inputs": {"conditioning": ["neg", 0], "reference_latents_method": "index_timestep_zero"}}
    if image_names:
        # The output canvas follows the first input image (aspect preserved, ~COMFY_MEGAPIXELS).
        g["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["scale0", 0], "vae": ["vae", 0]}}
    else:
        g["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    g["sample"] = {"class_type": "KSampler", "inputs": {
        "model": ["cfgnorm", 0], "seed": int(seed or 0), "steps": COMFY_STEPS, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["pos_ref", 0], "negative": ["neg_ref", 0],
        "latent_image": ["latent", 0], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    # PreviewImage writes to ComfyUI's temp dir (wiped on restart), not the user's output folder.
    g["out"] = {"class_type": "PreviewImage", "inputs": {"images": ["decode", 0]}}
    return g


def _comfy_torch_vram():
    devices = requests.get(f"{COMFY_URL}/system_stats", timeout=10).json()["devices"]
    return devices[0]["torch_vram_total"]


def _drop_opaque_alpha(png_bytes):
    """Qwen-Image-2.1 always decodes RGBA. Stage 5 treats the result as a 3-channel frame
    (`get_num_pixels_with_color` reshapes to 3), so an all-but-opaque alpha is dropped; a real
    cutout (cousin images asked for a transparent background) keeps its alpha for rembg."""
    image = Image.open(io.BytesIO(png_bytes))
    if image.mode != "RGBA":
        return png_bytes
    alpha = image.getchannel("A")
    if alpha.getextrema()[0] < 250:
        return png_bytes
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _qwen21_graph(prompt, image_names, seed):
    """Qwen-Image-2.1(-Turbo) edit graph, following Comfy-Org's image_qwen_image_2_1_image_edit template."""
    g = {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": Q21_UNET}},
        "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": "auto", "dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": Q21_CLIP, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": Q21_VAE}},
    }
    enc = {"clip": ["clip", 0], "prompt": prompt, "negative_prompt": "", "vae": ["vae", 0], "resolution": Q21_RESOLUTION}
    for i, name in enumerate(image_names[:16]):
        g[f"load{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc[f"images.image_{i + 1}"] = [f"load{i}", 0]
    g["enc"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}
    # The encoder's latent output is sized to the first reference image (or `resolution` without one).
    g["sample"] = {"class_type": "KSampler", "inputs": {
        "model": ["cache", 0], "seed": int(seed or 0), "steps": Q21_STEPS, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["enc", 0], "negative": ["enc", 1],
        "latent_image": ["enc", 2], "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["out"] = {"class_type": "PreviewImage", "inputs": {"images": ["decode", 0]}}
    return g


def comfy_free():
    """Ask ComfyUI to unload its models and wait until it has actually done so.

    /free only sets a flag that ComfyUI's worker acts on later; returning before that lets the caller
    move its own models back onto a GPU that is still full.
    """
    try:
        requests.post(f"{COMFY_URL}/free", json={"unload_models": True, "free_memory": True}, timeout=60)
    except requests.RequestException:
        return
    if not _wait_until(lambda: _comfy_torch_vram() < 512 * 2**20, timeout_s=120):
        print("[local-image] warning: ComfyUI still holds VRAM 120 s after /free", flush=True)


_TRANSPARENT_BG_RE = re.compile(r"transparent background", re.IGNORECASE)


def comfy_image_edit(prompt, images, seed=0, verbose=False):
    """Run Qwen-Image-Edit-2511 on up to three reference images; returns a list of PNG bytes."""
    t0 = time.time()
    if IMAGE_BACKEND == "qwen21":
        names = [_comfy_upload(b) for b in images[:16]]
        graph = _qwen21_graph(prompt, names, seed)
    else:
        # Asked for a "transparent background", Gemini returns a plain backdrop that rembg strips
        # later; Qwen-Image-Edit paints a literal checkerboard (and props) instead. Ask for what
        # rembg expects. (Qwen-Image-2.1 generates real transparency, so it gets the prompt as-is.)
        prompt = _TRANSPARENT_BG_RE.sub("plain pure white background with nothing else in it", prompt)
        names = [_comfy_upload(b) for b in images[:3]]
        graph = _qwen_edit_graph(prompt, names, seed)
    resp = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex}, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected the edit graph: {resp.text[:2000]}")
    prompt_id = resp.json()["prompt_id"]
    try:
        while True:
            hist = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=60).json()
            entry = hist.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI edit failed: {json.dumps(status.get('messages'))[:2000]}")
                if status.get("completed") or entry.get("outputs"):
                    break
            if time.time() - t0 > COMFY_TIMEOUT_S:
                raise TimeoutError(f"ComfyUI edit {prompt_id} exceeded {COMFY_TIMEOUT_S}s")
            time.sleep(1.0)
        out_images = []
        for img in entry["outputs"]["out"]["images"]:
            r = requests.get(f"{COMFY_URL}/view", params={"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                                          "type": img.get("type", "temp")}, timeout=120)
            r.raise_for_status()
            out_images.append(_drop_opaque_alpha(r.content))
    finally:
        if COMFY_FREE_AFTER:
            comfy_free()
    if verbose:
        print(f"[local-image] {'Qwen-Image-2.1-Turbo' if IMAGE_BACKEND == 'qwen21' else 'Qwen-Image-Edit-2511'}: {len(images)} ref image(s) -> {len(out_images)} image(s) "
              f"in {time.time() - t0:.1f}s", flush=True)
    return out_images


# ----------------------------------------------------------------------------------------------
# google.genai.Client look-alike
# ----------------------------------------------------------------------------------------------

OFFLOAD_GPU = os.environ.get("SIMFOUNDRY_LOCAL_OFFLOAD", "1") == "1"


def _is_module(obj, torch):
    try:
        return isinstance(obj, torch.nn.Module)
    except ReferenceError:
        return False


@contextlib.contextmanager
def gpu_released():
    """Park this process's CUDA-resident nn.Modules on the CPU while a local backend runs.

    Stage 5 holds SAM3 + DA3-giant + Prior-Depth-Anything (~12 GB) and then asks for an image edit;
    with Gemini that cost no VRAM, here ComfyUI/Ollama need most of the 12 GB card. Each module's own
    parameters/buffers are moved (recurse=False), so exactly the tensors that left are put back.
    """
    if not OFFLOAD_GPU:
        yield
        return
    try:
        import torch
    except ImportError:
        yield
        return
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        yield
        return
    import gc
    import warnings

    parked = []
    with warnings.catch_warnings():
        # isinstance() on deprecated proxies (e.g. torch.distributed.reduce_op) emits FutureWarning.
        warnings.simplefilter("ignore")
        modules = [o for o in gc.get_objects() if _is_module(o, torch)]
    for obj in modules:
        tensors = list(obj._parameters.values()) + list(obj._buffers.values())
        devices = {t.device for t in tensors if t is not None and t.is_cuda}
        if len(devices) == 1:
            parked.append((obj, devices.pop()))
    t0 = time.time()
    for module, _ in parked:
        module._apply(lambda t: t.cpu(), recurse=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if parked:
        print(f"[local-backend] parked {len(parked)} CUDA modules on CPU in {time.time() - t0:.1f}s "
              f"({torch.cuda.memory_reserved() / 2**30:.1f} GiB still reserved)", flush=True)
    try:
        yield
    finally:
        _restore_parked(parked, torch)


def _restore_parked(parked, torch, timeout_s=300):
    """Move parked modules back, waiting out other processes that are still releasing VRAM.

    A restore that gives up halfway leaves the model split across CPU and GPU, which only surfaces
    later as a device-mismatch error far from the cause; so this retries rather than raising early.
    """
    pending = list(parked)
    deadline = time.time() + timeout_s
    while pending:
        module, device = pending[0]
        try:
            module._apply(lambda t, d=device: t.to(d), recurse=False)
            pending.pop(0)
        except torch.OutOfMemoryError:
            if time.time() > deadline:
                raise
            torch.cuda.empty_cache()
            time.sleep(2.0)
    if parked:
        torch.cuda.synchronize()


_REPEATS = {}


def _effective_seed(model, prompt, images, seed):
    """Callers such as B1 retry with an unchanged seed and expect a fresh sample, which a remote
    model gives them and a seeded local one does not; bump the seed on each exact repeat."""
    key = (model, prompt, tuple(hash(b) for b in images), seed)
    n = _REPEATS.get(key, 0)
    _REPEATS[key] = n + 1
    return int(seed or 0) + 1000 * n


class _LocalModels:
    def __init__(self, verbose):
        self.verbose = verbose

    def generate_content(self, model, contents, config=None):
        prompt, images = _flatten_contents(contents)
        seed = _effective_seed(model, prompt, images, _config_value(config, "seed", 0))
        if _is_image_model(model):
            with gpu_released():
                out = comfy_image_edit(prompt, images, seed=seed, verbose=self.verbose)
            return _response(text=None, images=out)
        with gpu_released():
            text, finish = ollama_chat(
                prompt, images,
                temperature=_config_value(config, "temperature", 0.0),
                seed=seed,
                max_tokens=_config_value(config, "max_output_tokens", None),
                verbose=self.verbose,
            )
        if _wants_yx_order(prompt):
            text = to_gemini_coordinates(text)
        return _response(text=text, finish_reason=finish)

    def generate_content_stream(self, model, contents, config=None):
        yield self.generate_content(model=model, contents=contents, config=config)


class LocalGenAIClient:
    """Drop-in for ``google.genai.Client``; auth/http kwargs are accepted and ignored."""

    def __init__(self, *args, **kwargs):
        self.models = _LocalModels(verbose=os.environ.get("SIMFOUNDRY_LOCAL_VERBOSE", "1") == "1")


genai_module = SimpleNamespace(Client=LocalGenAIClient)
