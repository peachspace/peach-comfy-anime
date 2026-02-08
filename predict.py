import json
import os
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from cog import BasePredictor, Input, Path as CogPath


COMFY_DIR = Path("ComfyUI")
COMFY_INPUT_DIR = COMFY_DIR / "input"
WORKFLOW_PATH = Path("workflows") / "peach_comfy_anime_api.json"

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"


def _http_get(url: str, timeout: int = 10):
    return requests.get(url, timeout=timeout)


def _http_post(url: str, payload: dict, timeout: int = 30):
    return requests.post(url, json=payload, timeout=timeout)


def wait_for_comfyui(timeout_sec: int = 120):
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            r = _http_get(f"{BASE_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(f"ComfyUI did not start within {timeout_sec}s. Last error={last_err}")


def ensure_dirs():
    COMFY_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_blank_image(dst: Path, size=(512, 512)):
    img = Image.new("RGB", size, (128, 128, 128))
    img.save(dst)


def download_or_copy_image(src: str, dst: Path) -> Path:
    """
    src가 https URL이면 다운로드, 로컬 경로면 복사.
    ComfyUI가 읽을 수 있도록 ComfyUI/input 아래에 파일을 만들어줌.
    """
    ensure_dirs()

    if not src:
        raise ValueError("empty image source")

    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, stream=True, timeout=30)
        r.raise_for_status()

        # 확장자 힌트
        ct = (r.headers.get("content-type") or "").lower()
        if dst.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
            if "png" in ct:
                dst = dst.with_suffix(".png")
            elif "webp" in ct:
                dst = dst.with_suffix(".webp")
            else:
                dst = dst.with_suffix(".jpg")

        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return dst

    # 로컬 파일 경로
    p = Path(src).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"local image not found: {p}")
    shutil.copy(p, dst)
    return dst


def load_workflow() -> dict:
    wf_path = WORKFLOW_PATH
    if not wf_path.exists():
        raise FileNotFoundError(
            f"Workflow file not found: {wf_path}\n"
            f"Put your API JSON at: ~/peach-comfy-anime/{wf_path}"
        )
    with open(wf_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_first_node_id_by_class(workflow: dict, class_type: str) -> Optional[str]:
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return str(node_id)
    return None


def find_clip_nodes(workflow: dict):
    """
    CLIPTextEncode 노드들 중에서
    - negative로 보이는 텍스트( low quality / blurry / watermark 등 ) 가진 노드는 negative로,
    - 나머지 하나를 positive로 잡는다.
    """
    clip_nodes = []
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            text = (node.get("inputs", {}).get("text") or "").lower()
            clip_nodes.append((str(node_id), text))

    pos_id = None
    neg_id = None

    # negative 후보: 흔한 네거 키워드 포함된 노드 우선
    for nid, text in clip_nodes:
        if any(k in text for k in ["low quality", "worst quality", "blurry", "watermark", "signature", "jpeg artifacts"]):
            neg_id = nid
            break

    # positive 후보: negative가 아닌 것 중 첫 번째
    for nid, _ in clip_nodes:
        if nid != neg_id:
            pos_id = nid
            break

    # fallback: 둘 다 못 찾으면 그냥 첫/둘
    if pos_id is None and clip_nodes:
        pos_id = clip_nodes[0][0]
    if neg_id is None and len(clip_nodes) >= 2:
        neg_id = clip_nodes[1][0]

    return pos_id, neg_id


def update_workflow(
    workflow: dict,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    reference_image_path: Path,
    ip_adapter_scale: float,
):
    """
    네가 뽑아낸 API JSON(InstantID 기본 워크플로우)을 기준으로 “필수값만” 치환.
    (초보가 JSON/노드 ID를 직접 만질 필요 없게 여기서 알아서 바꾼다)
    """

    # 1) KSampler
    ks_id = find_first_node_id_by_class(workflow, "KSampler")
    if ks_id:
        ks = workflow[ks_id]["inputs"]
        ks["seed"] = int(seed)
        ks["steps"] = int(steps)
        ks["cfg"] = float(cfg)
        if sampler_name:
            ks["sampler_name"] = sampler_name
        if scheduler:
            ks["scheduler"] = scheduler

    # 2) EmptyLatentImage (width/height)
    lat_id = find_first_node_id_by_class(workflow, "EmptyLatentImage")
    if lat_id:
        lat = workflow[lat_id]["inputs"]
        lat["width"] = int(width)
        lat["height"] = int(height)

    # 3) CLIPTextEncode (positive/negative)
    pos_id, neg_id = find_clip_nodes(workflow)
    if pos_id:
        workflow[pos_id]["inputs"]["text"] = str(prompt)
    if neg_id:
        workflow[neg_id]["inputs"]["text"] = str(negative_prompt)

    # 4) LoadImage (reference face)
    li_id = find_first_node_id_by_class(workflow, "LoadImage")
    if li_id:
        # ComfyUI LoadImage는 "ComfyUI/input" 폴더 기준 파일명만 넣으면 됨
        workflow[li_id]["inputs"]["image"] = reference_image_path.name

    # 5) ApplyInstantID / ControlNetApplyAdvanced scale
    #    (InstantID 기본 워크플로우 기준)
    ai_id = find_first_node_id_by_class(workflow, "ApplyInstantID")
    if ai_id:
        workflow[ai_id]["inputs"]["weight"] = float(ip_adapter_scale)

    cn_id = find_first_node_id_by_class(workflow, "ControlNetApplyAdvanced")
    if cn_id:
        # instantid에서 strength도 너무 세면 스타일이 흔들리거나 얼굴이 깨질 수 있어
        # weight와 비슷한 수준으로 따라가게(상한을 둠)
        workflow[cn_id]["inputs"]["strength"] = float(min(max(ip_adapter_scale, 0.35), 0.85))

    return workflow


def submit_prompt(workflow: dict) -> str:
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    r = _http_post(f"{BASE_URL}/prompt", payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    pid = data.get("prompt_id")
    if not pid:
        raise RuntimeError(f"ComfyUI /prompt returned unexpected: {data}")
    return pid


def wait_for_result(prompt_id: str, timeout_sec: int = 300) -> dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        r = _http_get(f"{BASE_URL}/history/{prompt_id}", timeout=20)
        if r.status_code == 200:
            hist = r.json()
            # history 응답이 {prompt_id: {...}} 형태거나, 그냥 {...} 형태로 올 수 있음
            if isinstance(hist, dict) and prompt_id in hist:
                item = hist[prompt_id]
                if item and item.get("outputs"):
                    return item
            elif isinstance(hist, dict) and hist.get("outputs"):
                return hist
        time.sleep(1.0)
    raise TimeoutError("ComfyUI result timeout")


def extract_first_image_info(history_item: dict) -> Optional[dict]:
    outputs = history_item.get("outputs") or {}
    # outputs: { node_id: { "images":[{filename, subfolder, type}, ...], ... }, ... }
    for _, out in outputs.items():
        imgs = out.get("images")
        if imgs and isinstance(imgs, list):
            return imgs[0]
    return None


def fetch_image(image_info: dict, dst: Path):
    filename = image_info.get("filename")
    subfolder = image_info.get("subfolder", "")
    img_type = image_info.get("type", "output")
    if not filename:
        raise RuntimeError(f"Invalid image_info: {image_info}")

    params = f"filename={filename}&subfolder={subfolder}&type={img_type}"
    url = f"{BASE_URL}/view?{params}"
    r = _http_get(url, timeout=60)
    r.raise_for_status()
    with open(dst, "wb") as f:
        f.write(r.content)


class Predictor(BasePredictor):
    def setup(self):
        """
        Cog 컨테이너에서 1번만 실행됨.
        ComfyUI 서버가 떠있으면 그대로 쓰고, 아니면 여기서 띄운다.
        """
        ensure_dirs()

        # 이미 떠 있으면 재실행 안 함
        try:
            r = _http_get(f"{BASE_URL}/system_stats", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass

        # ComfyUI 실행
        cmd = [
            "python",
            str(COMFY_DIR / "main.py"),
            "--listen",
            COMFY_HOST,
            "--port",
            str(COMFY_PORT),
        ]
        # stdout/stderr는 Cog 로그로 남게
        self.proc = subprocess.Popen(cmd, cwd=str(Path(".")))
        wait_for_comfyui(timeout_sec=120)

    def predict(
        self,
        prompt: str = Input(description="Positive prompt"),
        negative_prompt: str = Input(description="Negative prompt", default="low quality, blurry, watermark, text"),
        width: int = Input(description="Width", default=896, ge=256, le=1536),
        height: int = Input(description="Height", default=1152, ge=256, le=1536),
        seed: int = Input(description="Seed (0=random)", default=0),
        steps: int = Input(description="Steps", default=30, ge=1, le=80),
        cfg: float = Input(description="CFG", default=7.0, ge=1.0, le=20.0),
        sampler_name: str = Input(description="Sampler", default="euler"),
        scheduler: str = Input(description="Scheduler", default="normal"),
        reference_image: str = Input(
            description="Reference image URL (https) or local path. If empty, a blank image is used.",
            default="",
        ),
        ip_adapter_scale: float = Input(
            description="Identity strength (InstantID/IPAdapter scale)",
            default=0.65,
            ge=0.0,
            le=1.2,
        ),
        mode: str = Input(
            description="mode hint (character/emotion/situation/main/background/event)",
            default="character",
        ),
    ) -> CogPath:
        """
        Firebase에서 보내는 input 형태를 그대로 맞춤.
        실제 동작은: workflow(API JSON) 불러오기 -> 값 치환 -> ComfyUI 실행 -> 첫 이미지 반환
        """

        # seed 처리: 0이면 랜덤
        if not seed or int(seed) == 0:
            seed = int.from_bytes(os.urandom(4), "big")

        # mode에 따라 ip 강도 기본값을 살짝 조정(원하면 여기 수치만 바꾸면 됨)
        m = (mode or "").lower().strip()
        if m == "emotion":
            ip = 0.55
        elif m == "situation":
            ip = 0.60
        elif m == "main":
            ip = 0.60
        elif m == "event":
            ip = 0.60
        else:
            ip = float(ip_adapter_scale)

        # reference 이미지 준비(없으면 blank로 대체해서 워크플로우가 안 죽게)
        ref_dst = COMFY_INPUT_DIR / f"ref_{uuid.uuid4().hex}.png"
        if reference_image.strip():
            try:
                download_or_copy_image(reference_image.strip(), ref_dst)
            except Exception as e:
                # 다운로드 실패하면 blank로라도 진행
                save_blank_image(ref_dst)
        else:
            save_blank_image(ref_dst)

        workflow = load_workflow()

        workflow = update_workflow(
            workflow=workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            reference_image_path=ref_dst,
            ip_adapter_scale=ip,
        )

        prompt_id = submit_prompt(workflow)
        hist_item = wait_for_result(prompt_id, timeout_sec=300)
        img_info = extract_first_image_info(hist_item)
        if not img_info:
            raise RuntimeError(f"No image output in history: {hist_item}")

        out_path = Path("/tmp") / f"out_{uuid.uuid4().hex}.png"
        fetch_image(img_info, out_path)

        return CogPath(str(out_path))
