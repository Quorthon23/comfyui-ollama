from __future__ import annotations
import copy
import json
import random
import re
import urllib.error
import urllib.parse
import urllib.request
import aiohttp

from ollama import Client
import numpy as np
import base64
from io import BytesIO
from server import PromptServer
from aiohttp import web
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
from typing import TYPE_CHECKING, Any, Literal
from dataclasses import dataclass, field
from pydantic.json_schema import JsonSchemaValue

# For type checking only. Torch is not installed at runtime
if TYPE_CHECKING:
    import torch


@dataclass
class ChatSession:
    messages: list[dict] = field(default_factory=list)
    model: str = ""


# Dictionary global per session_id
CHAT_SESSIONS: dict[str, ChatSession] = {}

# Function to filter enabled options
def _filter_enabled_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only the ollama options whose 'enable_*' flag is True."""
    if not options:
        return None
    enablers = [
        "enable_mirostat",
        "enable_mirostat_eta",
        "enable_mirostat_tau",
        "enable_num_ctx",
        "enable_repeat_last_n",
        "enable_repeat_penalty",
        "enable_temperature",
        "enable_seed",
        "enable_stop",
        "enable_tfs_z",
        "enable_num_predict",
        "enable_top_k",
        "enable_top_p",
        "enable_min_p",
    ]
    out: dict[str, Any] = {}
    for enabler in enablers:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            out[key] = options[key]
    return out or None


def _encode_images(images: Any, kwargs: dict[str, Any]) -> list[str] | None:
    """Helper to extract, flatten, and base64-encode image tensors/lists for Ollama."""
    raw_images: list[Any] = []

    def collect(item: Any):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                collect(sub)
        else:
            raw_images.append(item)

    collect(images)
    for k, v in kwargs.items():
        if k.startswith("image"):
            collect(v)

    if not raw_images:
        return None

    images_b64: list[str] = []
    for img_obj in raw_images:
        if hasattr(img_obj, "dim"):
            dim = img_obj.dim()
            if dim == 4:
                tensor_list = [img_obj[b] for b in range(img_obj.shape[0])]
            elif dim == 3:
                tensor_list = [img_obj]
            else:
                continue
        elif isinstance(img_obj, np.ndarray):
            if img_obj.ndim == 4:
                tensor_list = [img_obj[b] for b in range(img_obj.shape[0])]
            elif img_obj.ndim == 3:
                tensor_list = [img_obj]
            else:
                continue
        else:
            continue

        for img_tensor in tensor_list:
            if hasattr(img_tensor, "cpu"):
                array = img_tensor.cpu().numpy()
            else:
                array = img_tensor
            i = 255.0 * array
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = base64.b64encode(buffered.getvalue()).decode("utf-8")
            images_b64.append(img_bytes)

    return images_b64 if images_b64 else None


@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()

    url = data.get("url")
    client = Client(host=url)

    models = client.list().get('models', [])

    try:
        models = [model['model'] for model in models]
        return web.json_response(models)
    except Exception as e:
        models = [model['name'] for model in models]
        return web.json_response(models)


def _get_unsloth_auth_headers(api_key: str = "") -> dict[str, str]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        return headers
    try:
        db_path = os.path.expanduser("~/.unsloth/studio/auth/auth.db")
        if os.path.exists(db_path):
            import sqlite3
            import jwt
            from datetime import datetime, timezone, timedelta
            conn = sqlite3.connect(db_path)
            row = conn.execute("SELECT jwt_secret FROM auth_user WHERE username='unsloth'").fetchone()
            if row and row[0]:
                secret = row[0]
                payload = {
                    "sub": "unsloth",
                    "desktop": True,
                    "exp": datetime.now(timezone.utc) + timedelta(hours=2),
                }
                token = jwt.encode(payload, secret, algorithm="HS256")
                headers["Authorization"] = f"Bearer {token}"
    except Exception:
        pass
    return headers


@PromptServer.instance.routes.post("/unsloth/get_models")
async def unsloth_get_models_endpoint(request):
    data = await request.json()
    url = data.get("url", "http://127.0.0.1:8888").rstrip("/")
    api_key = data.get("api_key", "").strip()

    if not url.endswith("/v1"):
        endpoint = f"{url}/v1/models"
    else:
        endpoint = f"{url}/models"

    headers = _get_unsloth_auth_headers(api_key)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    res_data = await resp.json()
                    models_list = res_data.get("data", [])
                    models = [m.get("id") for m in models_list if isinstance(m, dict) and "id" in m]
                    return web.json_response(models)
                else:
                    err_text = await resp.text()
                    return web.json_response({"error": f"HTTP {resp.status}: {err_text}"}, status=resp.status)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/unsloth/get_variants")
async def unsloth_get_variants_endpoint(request):
    data = await request.json()
    url = data.get("url", "http://127.0.0.1:8888").rstrip("/")
    api_key = data.get("api_key", "").strip()
    model = data.get("model", "").strip()

    if not model:
        return web.json_response([])

    base_url = url[:-3] if url.endswith("/v1") else url
    headers = _get_unsloth_auth_headers(api_key)
    quoted_model = urllib.parse.quote(model, safe="")

    variants_data = None
    try:
        async with aiohttp.ClientSession() as session:
            # Query ONLY local cache on device (do not fetch remote undownloaded variants)
            local_ep = f"{base_url}/api/models/gguf-variants?repo_id={quoted_model}&prefer_local_cache=true"
            async with session.get(local_ep, headers=headers, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                if resp.status == 200:
                    variants_data = await resp.json()
    except Exception:
        pass

    if not variants_data or "variants" not in variants_data:
        return web.json_response([])

    raw_variants = variants_data.get("variants", [])
    downloaded = []
    for v in raw_variants:
        quant = v.get("quant")
        if not quant:
            continue
        if v.get("downloaded"):
            downloaded.append(quant)

    return web.json_response(downloaded)


@PromptServer.instance.routes.post("/unsloth/load_model")
async def unsloth_load_model_endpoint(request):
    data = await request.json()
    url = data.get("url", "http://127.0.0.1:8888").rstrip("/")
    api_key = data.get("api_key", "").strip()
    model = data.get("model", "").strip()
    quantization = data.get("quantization", "").strip()
    variant = quantization.split(" ")[0].strip() if quantization and quantization != "default" else ""
    context_length = data.get("context_length", 32768)

    if not model:
        return web.json_response({"error": "No model specified"}, status=400)

    if not url.endswith("/v1"):
        endpoint = f"{url}/v1/load"
    else:
        endpoint = f"{url}/load"

    headers = _get_unsloth_auth_headers(api_key)
    headers["Content-Type"] = "application/json"

    payload = {
        "model_path": model,
        "speculative_type": "off",
        "force_cancel_active": True,
    }
    if variant:
        payload["gguf_variant"] = variant
    if context_length and int(context_length) > 0:
        payload["max_seq_length"] = int(context_length)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                if resp.status == 200:
                    res_text = await resp.text()
                    res_data = json.loads(res_text.strip())
                    if "_deferred_error" in res_data:
                        err_info = res_data["_deferred_error"]
                        detail = err_info.get("detail", str(err_info))
                        return web.json_response({"error": f"Load error: {detail}"}, status=500)
                    return web.json_response({"status": "ok", "data": res_data})
                else:
                    err_text = await resp.text()
                    return web.json_response({"error": f"HTTP {resp.status}: {err_text}"}, status=resp.status)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@PromptServer.instance.routes.post("/unsloth/unload_model")
async def unsloth_unload_model_endpoint(request):
    data = await request.json()
    url = data.get("url", "http://127.0.0.1:8888").rstrip("/")
    api_key = data.get("api_key", "").strip()
    model = data.get("model", "").strip()

    if not model:
        return web.json_response({"error": "No model specified"}, status=400)

    if not url.endswith("/v1"):
        endpoint = f"{url}/v1/unload"
    else:
        endpoint = f"{url}/unload"

    headers = _get_unsloth_auth_headers(api_key)
    headers["Content-Type"] = "application/json"

    payload = {
        "model_path": model,
        "force_cancel_active": True,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    res_text = await resp.text()
                    res_data = json.loads(res_text.strip())
                    return web.json_response({"status": "ok", "data": res_data})
                else:
                    err_text = await resp.text()
                    return web.json_response({"error": f"HTTP {resp.status}: {err_text}"}, status=resp.status)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


def _unsloth_configure_auto_unload(
    url: str,
    api_key: str,
    keep_alive: int,
    keep_alive_unit: str = "minutes",
    debug: bool = False,
) -> None:
    """Configure Unsloth server-side idle auto-unload timeout (PUT /api/settings/openai-auto-switch)."""
    base_url = url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    endpoint = f"{base_url}/api/settings/openai-auto-switch"

    headers = _get_unsloth_auth_headers(api_key)
    headers["Content-Type"] = "application/json"

    if keep_alive > 0:
        idle_seconds = keep_alive * 60 if keep_alive_unit == "minutes" else keep_alive * 3600
        payload = {
            "enabled": True,
            "auto_unload_idle_seconds": idle_seconds,
            "auto_unload_api_only": False,
        }
    elif keep_alive == 0:
        return
    else:
        payload = {
            "enabled": False,
            "auto_unload_idle_seconds": 0,
            "auto_unload_api_only": False,
        }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="PUT"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if debug:
                print("[Unsloth] Auto-unload configuration updated:", data)
    except Exception as e:
        if debug:
            print(f"[Unsloth Auto-Unload Warning]: Could not update auto-switch setting: {e}")


def _unsloth_unload_model(
    url: str,
    api_key: str,
    model: str,
    timeout: int = 60,
    debug: bool = False,
) -> None:
    """Explicitly unload model from memory via POST /v1/unload."""
    if not model:
        return

    base_url = url.rstrip("/")
    if not base_url.endswith("/v1"):
        endpoint = f"{base_url}/v1/unload"
    else:
        endpoint = f"{base_url}/unload"

    headers = _get_unsloth_auth_headers(api_key)
    headers["Content-Type"] = "application/json"

    payload = {
        "model_path": model,
        "force_cancel_active": True,
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_bytes = resp.read()
            data = json.loads(resp_bytes.decode("utf-8").strip())
            if debug:
                print(f"[Unsloth] Model '{model}' unload response:", data)
            print(f"[Unsloth] Model '{model}' unloaded from memory.")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return
        if debug:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"[Unsloth Unload Warning {e.code}]: {err_body}")
    except Exception as e:
        if debug:
            print(f"[Unsloth Unload Warning]: {str(e)}")


def _unsloth_ensure_model_loaded(
    url: str,
    api_key: str,
    model: str,
    variant: str = "",
    context_length: int = 32768,
    debug: bool = False,
) -> None:
    """Ensure requested model is loaded into VRAM/memory via POST /v1/load before inference."""
    if not model:
        return

    base_url = url.rstrip("/")
    if not base_url.endswith("/v1"):
        endpoint = f"{base_url}/v1/load"
    else:
        endpoint = f"{base_url}/load"

    headers = _get_unsloth_auth_headers(api_key)
    headers["Content-Type"] = "application/json"

    payload: dict[str, Any] = {
        "model_path": model,
        "speculative_type": "off",
        "force_cancel_active": True,
    }
    if variant:
        payload["gguf_variant"] = variant
    if context_length and int(context_length) > 0:
        payload["max_seq_length"] = int(context_length)

    variant_info = f" ({variant})" if variant else ""
    ctx_info = f", context: {context_length}" if context_length else ""
    print(f"[Unsloth] Checking / loading model '{model}'{variant_info}{ctx_info} into memory...")
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=None) as resp:
            resp_bytes = resp.read()
            # Unsloth pads responses with whitespace while loading
            data = json.loads(resp_bytes.decode("utf-8").strip())
            if "_deferred_error" in data:
                err_info = data["_deferred_error"]
                detail = err_info.get("detail", str(err_info))
                raise Exception(f"[Unsloth Load Error]: {detail}")
            if debug:
                print(f"[Unsloth] Model '{model}' load response:", data)
            print(f"[Unsloth] Model '{model}'{variant_info} is loaded and ready.")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Server does not support /v1/load (e.g. generic OpenAI or other backend), ignore
            return
        err_body = e.read().decode("utf-8", errors="replace")
        raise Exception(f"[Unsloth Load Error {e.code}]: {err_body}")
    except Exception as e:
        raise e


def _unsloth_chat_completion(
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    variant: str = "",
    context_length: int = 32768,
    request_options: dict[str, Any] | None = None,
    response_format: str | None = None,
    think: bool = False,
    debug: bool = False,
) -> tuple[str, str | None]:
    # 1. Automatically ensure model is loaded in memory first with specified context length
    _unsloth_ensure_model_loaded(url, api_key, model, variant=variant, context_length=context_length, debug=debug)

    base_url = url.rstrip("/")
    if not base_url.endswith("/v1"):
        endpoint = f"{base_url}/v1/chat/completions"
    else:
        endpoint = f"{base_url}/chat/completions"

    headers = _get_unsloth_auth_headers(api_key)
    headers["Content-Type"] = "application/json"

    # Use streaming for real-time progress and keep-alive stability during generation
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}

    if request_options:
        if "temperature" in request_options:
            payload["temperature"] = request_options["temperature"]
        if "top_p" in request_options:
            payload["top_p"] = request_options["top_p"]
        if "seed" in request_options:
            payload["seed"] = request_options["seed"]
        if "num_predict" in request_options and request_options["num_predict"] > 0:
            payload["max_tokens"] = request_options["num_predict"]
        if "stop" in request_options and request_options["stop"]:
            payload["stop"] = request_options["stop"]
        if "top_k" in request_options:
            payload["top_k"] = request_options["top_k"]
        if "min_p" in request_options:
            payload["min_p"] = request_options["min_p"]
        if "repeat_penalty" in request_options:
            payload["repeat_penalty"] = request_options["repeat_penalty"]

    if debug:
        print(f"\n--- Unsloth Chat Completion Request:\nURL: {endpoint}\nModel: {model}\nPayload:\n")
        pprint(payload)
        print("---------------------------------------------------------")

    variant_info = f" ({variant})" if variant else ""
    print(f"[Unsloth] Waiting for response from '{model}'{variant_info}...")

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    try:
        with urllib.request.urlopen(req, timeout=None) as resp:
            content_type = resp.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            c = delta.get("content")
                            if c:
                                content_parts.append(c)
                            r = delta.get("reasoning_content")
                            if r:
                                reasoning_parts.append(r)
                content = "".join(content_parts)
                reasoning = "".join(reasoning_parts) if reasoning_parts else None
            else:
                # Non-streaming JSON fallback
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if not choices:
                    raise Exception(f"[Unsloth API] Empty choices returned: {data}")
                message = choices[0].get("message", {})
                content = message.get("content", "") or ""
                reasoning = message.get("reasoning_content")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise Exception(f"[Unsloth API Error {e.code}]: {error_body}")
    except Exception as e:
        raise Exception(f"[Unsloth Connection Error]: {str(e)}")

    if debug:
        print("\n--- Unsloth Chat Completion Response:\n")
        print(content)
        if reasoning:
            print(f"--- Thinking:\n{reasoning}")
        print("---------------------------------------------------------")

    print(f"[Unsloth] Generation complete ({len(content)} chars received).")

    if not reasoning and ("<think>" in content or "<thinking>" in content):
        match = re.search(r"<(?:think|thinking)>(.*?)</(?:think|thinking)>", content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            reasoning = match.group(1).strip()
            content = re.sub(r"<(?:think|thinking)>.*?</(?:think|thinking)>\s*", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

    thinking = reasoning if think else None
    return content, thinking

class OllamaSaveContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"context": ("STRING", {"forceInput": True},),
                     "filename": ("STRING", {"default": "context"})},
                }

    RETURN_TYPES = ()
    FUNCTION = "ollama_save_context"

    OUTPUT_NODE = True
    CATEGORY = "Ollama"

    def ollama_save_context(self, filename, context=None):
        path = self._base_dir + os.path.sep + filename
        metadata = PngInfo()

        metadata.add_text("context", ','.join(map(str, context)))

        image = Image.new('RGB', (100, 100), (255, 255, 255))  # Creates a 100x100 white image

        image.save(path + ".png", pnginfo=metadata)

        return {"ui": {"context": context}}


class OllamaLoadContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f != ".keep"]
        return {"required":
                    {"context_file": (files, {})},
                }

    CATEGORY = "Ollama"

    RETURN_NAMES = ("context",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ollama_load_context"

    def ollama_load_context(self, context_file):
        with Image.open(self._base_dir + os.path.sep + context_file) as img:
            info = img.info
            res = info.get('context', '')
        return (res,)


class OllamaOptionsV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_mirostat": ("BOOLEAN", {"default": False}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max":2, "step": 1, "tooltip": "Whether to use Mirostat sampling. Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. (0 = disabled, 1 = Mirostat 1, 2 = Mirostat 2.0)"}),

                "enable_mirostat_eta": ("BOOLEAN", {"default": False}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0, "step": 0.1, "tooltip": "Mirostat's learning rate parameter influences how quickly the algorithm responds to feedback from the generated text."}),

                "enable_mirostat_tau": ("BOOLEAN", {"default": False}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0, "step": 0.1, "tooltip": "Mirostat's target entropy parameter controls the balance between coherence and diversity in the generated text."}),

                "enable_num_ctx": ("BOOLEAN", {"default": False}),
                "num_ctx": ("INT", {"default": 2048, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the size of the context window used to generate the next token."}),

                "enable_repeat_last_n": ("BOOLEAN", {"default": False}),
                "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1, "tooltip": "Sets how far back for the model to look back to prevent repetition. (0 = disabled, -1 = num_ctx)"}),

                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05, "tooltip": "Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient."}),

                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": -10, "max": 10, "step": 0.05, "tooltip": "Increasing the temperature will make the model answer more creatively."}),

                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."}),

                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "When this pattern is encountered the LLM will stop generating text and return."}),

                "enable_tfs_z": ("BOOLEAN", {"default": False}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),

                "enable_num_predict": ("BOOLEAN", {"default": False}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1, "tooltip": "Maximum number of tokens to predict when generating text. The default -1 means infinite generation."}),

                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1, "tooltip": "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative."}),

                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05, "tooltip": "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text."}),

                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05, "tooltip": "Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out."}),

                "debug": ("BOOLEAN", {"default": False, "tooltip": "For debugging purposes of the custom nodes, no effect on ollama api."}),
            },
        }

    RETURN_TYPES = ("OLLAMA_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "ollama_options"
    CATEGORY = "Ollama"
    DESCRIPTION = "Various settings for advanced configuration of Ollama inference. See Ollama documentation for more details."

    def ollama_options(self, **kargs):

        if kargs['debug']:
            print("--- ollama options v2 dump\n")
            pprint(kargs)
            print("---------------------------------------------------------")

        return (kargs,)

class OllamaConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "The URL of the Ollama server. Default value points to a local instance with ollama's default port configuration."
                }),
                "model": ((), {"tooltip": "Select a model for inference. This is a list of available models on the Ollama server. If you don't see any, make sure the Ollama server is running on the url and there are models installed."}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1, "tooltip": "Configures how long ollama keeps the model loaded in memory after inference. -1 = keep alive indefinitely, 0 = unload model immediately after inference"}),
                "keep_alive_unit": (["minutes", "hours"],),
            },
        }

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "ollama_connectivity"
    CATEGORY = "Ollama"
    DESCRIPTION = "Provides connection to an Ollama server. Use the refresh button to load the model list in case of connection error or after installing a new model."

    def ollama_connectivity(self, url, model, keep_alive, keep_alive_unit):
        data = {
            "provider": "ollama",
            "url": url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
        }

        return (data,)


class UnslothConnectivity:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:8888",
                    "tooltip": "The URL of the Unsloth server (default http://127.0.0.1:8888 or http://127.0.0.1:8000)."
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "API Key from Unsloth Settings -> API (starts with sk-unsloth-...). Leave empty if no authentication is configured."
                }),
                "model": ((), {"tooltip": "Select a model loaded in Unsloth. Use the '🔄 Reconnect' button to refresh the list of available models."}),
                "quantization": ((), {"tooltip": "Select quantization/variant (e.g. UD-Q4_K_XL). Displays only quantizations already downloaded to device."}),
                "context_length": ("INT", {"default": 32768, "min": 0, "max": 262144, "step": 1024, "tooltip": "Context window size in tokens (default 32768 = 32K). Set 0 to use model default."}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1, "tooltip": "Configures how long Unsloth keeps the model loaded in memory after inference. -1 = keep alive indefinitely, 0 = unload model immediately after inference"}),
                "keep_alive_unit": (["minutes", "hours"],),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "unsloth_connectivity"
    CATEGORY = "Ollama"
    DESCRIPTION = "Provides connection to an Unsloth API server (OpenAI-compatible). Use the Reconnect button to load available models and quantizations."

    def unsloth_connectivity(self, url, api_key, model, quantization="", context_length=32768, keep_alive=5, keep_alive_unit="minutes", **kwargs):
        clean_variant = ""
        if quantization and quantization not in ("default", "(none)", "(not downloaded)", "none"):
            clean_variant = quantization.split(" ")[0].strip()
        data = {
            "provider": "unsloth",
            "url": url,
            "api_key": api_key,
            "model": model,
            "quantization": clean_variant,
            "context_length": context_length,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
        }

        return (data,)


class OllamaGenerateV2:
    def __init__(self):
        self.saved_context = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist.",
                    "tooltip": "System prompt - use this to set the role and general behavior of the model."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?",
                    "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'"
                }),
                "think": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail."}),
                "keep_context": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will keep the context of the conversation and use it for the next generation. This is useful for multi-turn conversations or tasks that require context."}),
                "format": (["text", "json"], {"tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output."}),

            },
            "optional": {
                "connectivity": ("OLLAMA_CONNECTIVITY", {"forceInput": False, "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set."},),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False, "tooltip": "Connect an Ollama Options node for advanced inference configuration."},),
                "images": ("COMFY_AUTOGROW_V3", {
                    "template": {
                        "input": {"required": {"image": ("IMAGE", {"tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response."})}},
                        "names": [f"image_{i}" for i in range(1, 40)],
                        "min": 0
                    }
                }),
                "context": ("OLLAMA_CONTEXT", {"forceInput": False, "tooltip": "Optionally set an existing model context, useful for multi-turn conversations, follow-up questions."},),
                "meta": ("OLLAMA_META", {"forceInput": False, "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along."},),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "OLLAMA_CONTEXT", "OLLAMA_META",)
    RETURN_NAMES = ("result", "thinking", "context", "meta",)
    FUNCTION = "ollama_generate_v2"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def get_request_options(self, options):
        response = None

        if options is None:
            return response

        enablers = ['enable_mirostat', 'enable_mirostat_eta',
                    'enable_mirostat_tau', 'enable_mirostat_eta',
                    'enable_num_ctx', 'enable_repeat_last_n', 'enable_repeat_penalty',
                    'enable_temperature', 'enable_seed', 'enable_stop', 'enable_tfs_z', 'enable_num_predict',
                    'enable_top_k', 'enable_top_p', 'enable_min_p']

        for enabler in enablers:
            if options[enabler]:
                if response is None:
                    response = {}
                key = enabler.replace("enable_", "")
                response[key] = options[key]

        return response

    def ollama_generate_v2(self, system, prompt, think, keep_context, format, context = None, options=None, connectivity=None, images=None, meta=None, **kwargs):

        if connectivity is None and meta is None:
            raise Exception("Required input connectivity or meta.")

        if connectivity is None and meta['connectivity'] is None:
            raise Exception("Required input connectivity or connectivity in meta.")

        if meta is not None:
            if connectivity is not None: # bypass the current meta connectivity
                meta["connectivity"] = connectivity
            if options is not None: # bypass the current meta options
                meta["options"] = options
        else:
            meta = {"options": options, "connectivity": connectivity}

        conn = meta['connectivity']
        provider = conn.get('provider', 'ollama')
        url = conn['url']
        model = conn['model']

        debug_print = True if meta['options'] is not None and meta['options']['debug'] else False

        if format == "text":
            format = ''

        images_b64 = _encode_images(images, kwargs)
        request_options = self.get_request_options(options)

        if provider == "unsloth":
            api_key = conn.get("api_key", "")
            variant = conn.get("quantization", "")
            context_length = conn.get("context_length", 32768)
            if request_options and "num_ctx" in request_options and request_options["num_ctx"] > 0:
                context_length = request_options["num_ctx"]
            keep_alive = conn.get("keep_alive", 5)
            keep_alive_unit = conn.get("keep_alive_unit", "minutes")

            # Configure server idle auto-unload
            _unsloth_configure_auto_unload(url, api_key, keep_alive, keep_alive_unit, debug=debug_print)

            # Build user content
            if images_b64:
                user_content: Any = [{"type": "text", "text": prompt}]
                for img_b64 in images_b64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
            else:
                user_content = prompt

            user_msg = {"role": "user", "content": user_content}

            # Handle context and conversation history
            if keep_context and context is None and isinstance(self.saved_context, list):
                messages = list(self.saved_context)
                if system:
                    if messages and messages[0].get("role") == "system":
                        messages[0] = {"role": "system", "content": system}
                    else:
                        messages.insert(0, {"role": "system", "content": system})
                messages.append(user_msg)
            else:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                if context is not None:
                    if isinstance(context, list) and all(isinstance(m, dict) for m in context):
                        messages.extend(context)
                    elif isinstance(context, str):
                        try:
                            parsed = json.loads(context)
                            if isinstance(parsed, list):
                                messages.extend(parsed)
                        except Exception:
                            pass
                messages.append(user_msg)

            if debug_print:
                print(f"""
--- unsloth generate request: 
url: {url}
model: {model}
variant: {variant}
context_length: {context_length}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
think: {think}
options: {request_options}
keep alive: {keep_alive} {keep_alive_unit}
format: {format}
---------------------------------------------------------
""")

            variant_info = f" ({variant})" if variant else ""
            print(f"[Unsloth] Generate: requesting '{model}'{variant_info} with {0 if images_b64 is None else len(images_b64)} image(s)...")
            res_text, res_thinking = _unsloth_chat_completion(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                variant=variant,
                context_length=context_length,
                request_options=request_options,
                response_format=format,
                think=think,
                debug=debug_print,
            )

            # If keep_alive == 0, immediately unload the model from memory
            if keep_alive == 0:
                _unsloth_unload_model(url, api_key, model, debug=debug_print)

            # Update context
            messages.append({"role": "assistant", "content": res_text})
            if keep_context:
                self.saved_context = messages
                if debug_print:
                    print("saving context to node memory.")

            return res_text, res_thinking, messages, meta

        # Ollama provider branch
        client = Client(host=url)

        if context is not None and isinstance(context, str):
            string_list = context.split(',')
            context = [int(item.strip()) for item in string_list if item.strip().isdigit()]

        if keep_context and context is None:
            context = self.saved_context

        keep_alive_unit = 'm' if conn.get('keep_alive_unit') == "minutes" else 'h'
        request_keep_alive = str(conn.get('keep_alive', 5)) + keep_alive_unit

        if debug_print:
            print(f"""
--- ollama generate v2 request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
context: {context}
think: {think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
---------------------------------------------------------
""")

        print(f"[Ollama] Generate: requesting '{model}' with {0 if images_b64 is None else len(images_b64)} image(s)...")
        response = client.generate(
            model=model,
            system=system,
            prompt=prompt,
            images=images_b64,
            context=context,
            think=think,
            options=request_options,
            keep_alive=request_keep_alive,
            format=format,
        )

        if debug_print:
            print("\n--- ollama generate v2 response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response['response']
        ollama_response_thinking = response['thinking'] if think else None

        if keep_context:
            self.saved_context = response["context"]
            if debug_print:
                print("saving context to node memory.")

        return ollama_response_text, ollama_response_thinking, response['context'], meta,


class OllamaChat:
    """
    Text generation with Ollama Chat.
    Returns: (result: str, thinking: str|None, meta: dict, history: str)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an AI artist.",
                        "tooltip": "System prompt - use this to set the role and general behavior of the model.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What is art?",
                        "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'",
                    },
                ),
                "think": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail.",
                    },
                ),
                "format": (
                    ["text", "json"],
                    {
                        "tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output."
                    },
                ),
            },
            "optional": {
                "connectivity": (
                    "OLLAMA_CONNECTIVITY",
                    {
                        "forceInput": False,
                        "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set.",
                    },
                ),
                "options": (
                    "OLLAMA_OPTIONS",
                    {
                        "forceInput": False,
                        "tooltip": "Connect an Ollama Options node for advanced inference configuration.",
                    },
                ),
                "images": (
                    "COMFY_AUTOGROW_V3",
                    {
                        "template": {
                            "input": {"required": {"image": ("IMAGE", {"tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response."})}},
                            "names": [f"image_{i}" for i in range(1, 40)],
                            "min": 0
                        }
                    },
                ),
                "meta": (
                    "OLLAMA_META",
                    {
                        "forceInput": False,
                        "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along.",
                    },
                ),
                "history": (
                    "OLLAMA_HISTORY",
                    {
                        "forceInput": False,
                        "tooltip": "Optionally set an existing model history, useful for multi-turn conversations, follow-up questions.",
                    },
                ),
                "reset_session": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear the conversation history. WARNING: If using shared history, this will affect all nodes using the same history ID.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "OLLAMA_META",
        "OLLAMA_HISTORY",
    )
    RETURN_NAMES = (
        "result",
        "thinking",
        "meta",
        "history",
    )
    FUNCTION = "ollama_chat"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama Chat. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    def ollama_chat(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        options: dict[str, Any] | None = None,
        connectivity: dict[str, Any] | None = None,
        images: list[torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
        history: str | None = None,
        reset_session: bool = False,
        **kwargs
    ) -> tuple[str | None, str | None, dict[str, Any], str | None]:

        if meta is None:
            if connectivity is None:
                raise ValueError("Either 'connectivity' or 'meta' must be provided.")
            meta = {}

        # Update with provided values (override)
        if connectivity is not None:
            meta["connectivity"] = connectivity
        if options is not None:
            meta["options"] = options
        else:
            meta["options"] = None

        # Final validation
        if "connectivity" not in meta or meta["connectivity"] is None:
            raise ValueError("'connectivity' must be present in meta.")

        url = meta["connectivity"]["url"]
        model = meta["connectivity"]["model"]
        client = Client(host=url)

        debug_print = (
            True if meta["options"] is not None and meta["options"]["debug"] else False
        )

        ollama_format: Literal["", "json"] | JsonSchemaValue | None = None

        if format == "json":
            ollama_format = "json"
        elif format == "text":
            ollama_format = ""

        keep_alive_unit = (
            "m" if meta["connectivity"]["keep_alive_unit"] == "minutes" else "h"
        )
        request_keep_alive = str(meta["connectivity"]["keep_alive"]) + keep_alive_unit

        # 4. use the shared helper instead of self.get_request_options
        request_options = _filter_enabled_options(options)

        images_b64 = _encode_images(images, kwargs)

        if debug_print:
            print(
                f"""
--- ollama chat request: 

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
think: {think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
---------------------------------------------------------
"""
            )

        # Determinate which session to use
        session_key = history if history is not None else unique_id

        # If reset_session is True, reset the session
        if reset_session:
            CHAT_SESSIONS[session_key] = ChatSession()
            if debug_print:
                print(f"Session {session_key} has been reset")

        # If the session doesn't exist, create it
        if session_key not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_key] = ChatSession()

        session = CHAT_SESSIONS[session_key]

        # Update history for return
        history = session_key

        # If there is a system prompt, replace it or add it to the beginning
        if system:
            if session.messages and session.messages[0].get("role") == "system":
                session.messages[0] = {"role": "system", "content": system}
            else:
                session.messages.insert(0, {"role": "system", "content": system})

        # Construct the user message for history
        user_message_for_history: dict[str, Any] = {
            "role": "user",
            "content": prompt,
        }

        # Add the user message to the history (without images)
        session.messages.append(user_message_for_history)

        if debug_print:
            print("\n--- ollama chat session:")
            for message in session.messages:
                pprint(f"{message['role']}> {message['content'][:50]}...")
                if "images" in message:
                    for image in message["images"]:
                        pprint(f"Image: {image[:50]}...")
            print("---------------------------------------------------------")

        conn = meta['connectivity']
        provider = conn.get('provider', 'ollama')

        # Construct the messages for the API call (with images)
        messages_for_api = copy.deepcopy(session.messages)

        if provider == "unsloth":
            api_key = conn.get("api_key", "")
            variant = conn.get("quantization", "")
            context_length = conn.get("context_length", 32768)
            if request_options and "num_ctx" in request_options and request_options["num_ctx"] > 0:
                context_length = request_options["num_ctx"]
            keep_alive = conn.get("keep_alive", 5)
            keep_alive_unit = conn.get("keep_alive_unit", "minutes")

            # Configure server idle auto-unload
            _unsloth_configure_auto_unload(url, api_key, keep_alive, keep_alive_unit, debug=debug_print)

            if images_b64 is not None:
                content_parts = [{"type": "text", "text": prompt}]
                for img_b64 in images_b64:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })
                messages_for_api[-1]["content"] = content_parts

            variant_info = f" ({variant})" if variant else ""
            print(f"[Unsloth] Chat: requesting '{model}'{variant_info} with {0 if images_b64 is None else len(images_b64)} image(s)...")
            res_text, res_thinking = _unsloth_chat_completion(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages_for_api,
                variant=variant,
                context_length=context_length,
                request_options=request_options,
                response_format=format,
                think=think,
                debug=debug_print,
            )

            # If keep_alive == 0, immediately unload the model from memory
            if keep_alive == 0:
                _unsloth_unload_model(url, api_key, model, debug=debug_print)

            session.messages.append({
                "role": "assistant",
                "content": res_text,
            })

            return (
                res_text,
                res_thinking,
                meta,
                history,
            )

        # If there are images, modify the last user message for the API call
        if images_b64 is not None:
            messages_for_api[-1]["images"] = images_b64

        print(f"[Ollama] Chat: requesting '{model}' with {0 if images_b64 is None else len(images_b64)} image(s)...")
        response = client.chat(

            model=model,
            messages=messages_for_api,
            options=request_options,
            keep_alive=request_keep_alive,
            format=ollama_format,
        )

        if debug_print:
            print("\n--- ollama chat response:")
            pprint(response)
            print("---------------------------------------------------------")

        ollama_response_text = response.message.content
        ollama_response_thinking = response.message.thinking if think else None

        # Add the assistant message to the history
        session.messages.append(
            {
                "role": "assistant",
                "content": ollama_response_text,
            }
        )

        return (
            ollama_response_text,
            ollama_response_thinking,
            meta,
            history,
        )


NODE_CLASS_MAPPINGS = {
    "OllamaOptionsV2": OllamaOptionsV2,
    "OllamaConnectivityV2": OllamaConnectivityV2,
    "UnslothConnectivity": UnslothConnectivity,
    "OllamaGenerateV2": OllamaGenerateV2,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
    "OllamaChat": OllamaChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaOptionsV2": "Ollama Options",
    "OllamaConnectivityV2": "Ollama Connectivity",
    "UnslothConnectivity": "Unsloth Connectivity",
    "OllamaGenerateV2": "Ollama Generate",
    "OllamaSaveContext": "Ollama Save Context",
    "OllamaLoadContext": "Ollama Load Context",
    "OllamaChat": "Ollama Chat",
}
