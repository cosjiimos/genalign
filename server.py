from __future__ import annotations
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops
import torch, os, re, base64, uuid
from contextlib import nullcontext
from io import BytesIO
from math import ceil
import cv2 , numpy as np
import base64, io, os
import json
import datetime
import tempfile
from typing import List, Dict
# ───────── Semantic-SAM 의존 코드 ─────────
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
from tasks import interactive_infer_image_idino_m2m_auto
from utils.constants import COCO_PANOPTIC_CLASSES
# ──────────────────────────────────────────
import server_utils as util  
import requests, shutil  
from openai import OpenAI
from flask import send_from_directory
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
import numpy as np
from pathlib import Path 
import csv
import traceback
app = Flask(__name__)
CORS(app)
# ------------------ 장치 설정 ------------------
DEVICE  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AMP_CTX = (
    torch.autocast(device_type="cuda:0", dtype=torch.float16)
    if DEVICE.type == "cuda:0" else nullcontext()
)

# ======================= ① 전역로드 =========================
MODEL_DIR = "./models/clip-vit-base-patch32"
os.environ["HF_HUB_OFFLINE"] = "1"   # 완전 오프라인 모드

clip_model = CLIPModel.from_pretrained(MODEL_DIR).to(DEVICE)  # ← 여기!
clip_proc  = CLIPProcessor.from_pretrained(MODEL_DIR)

faiss_index = faiss.read_index("clip_image_index.index")

with open("clip_image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)                 # ↔ index 의 id 와 1:1 리스트
    
# ───────── 모델 로딩 (GPU) ─────────
config_path = "configs/semantic_sam_only_sa-1b_swinL.yaml"
ckpt_path   = "checkpoints/swinl_only_sam_many2many.pth"
opt         = load_opt_from_config_file(config_path)
model       = BaseModel(opt, build_model(opt)).from_pretrained(ckpt_path).eval().to(DEVICE)

# ──────────────────────────────────────────
SESSION_ROOT = Path("logs")   # logs/<user>/<version>/  구조로 생성
GENERATED_DIR = os.path.join("client", "public", "img", "sofa_generated")
bg_removed_path = os.path.join("client", "public", "img", "removed_bg")

# Set OpenAI API key from environment variable or config
if "OPENAI_API_KEY" not in os.environ:
    try:
        from config import OPENAI_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except ImportError:
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment or config.py")
        print("   Please set: export OPENAI_API_KEY='your-key-here'")
        print("   Or create config.py from config.example.py")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
current_session = {"user": "anonymous", "version": "default"}

# 클래스·파츠 목록
all_classes = []
all_parts   = []
prompt_level = [3,4,5, 6]
IGNORED_LABELS = {"background", "shadow", "floor", "wall", "ceiling"}
TEXT_SIZE = 640          # ↔ 클라이언트 호출과 반드시 동일




@app.route("/segment", methods=["POST"])
def segment():
    data = request.get_json(silent=True) or {}
    user    = (data.get("user")    or current_session["user"]).strip()
    version = (data.get("version") or current_session["version"]).strip()
    
    # 1) Base64 → PIL.Image (EXIF 방향 보정)
    img_b64 = data.get("image_b64")
    if not img_b64:
        return jsonify(error="no image_b64"), 400

    try:
        img_bytes  = base64.b64decode(img_b64)
        image_orig = ImageOps.exif_transpose(Image.open(BytesIO(img_bytes))).convert("RGB")
        fg_rgba, fg_mask = util.remove_background(image_orig, smooth_edges=False)
        
        bg_removed_name = f"nobg_{uuid.uuid4().hex}.png"
        bg_removed_path = os.path.join("client", "public", "img", "nobg", bg_removed_name)
        os.makedirs(os.path.dirname(bg_removed_path), exist_ok=True)
        fg_rgba.save(bg_removed_path)          # ← 투명 배경 RGBA 저장
    except Exception as e:
        return jsonify(error=f"decode fail: {e}"), 400
    
    # --- NEW: 투명 RGBA → 흰 배경 RGB 컴포지트 ---
    canvas = Image.new("RGB", fg_rgba.size, (255,255,255))
    alpha  = fg_rgba.split()[-1]
    canvas.paste(fg_rgba, mask=alpha)
    image_for_seg = canvas
    orig_w, orig_h = image_for_seg.size
    
        # 1) Semantic-SAM 추론 → 전경만 입력
    with torch.no_grad(), AMP_CTX:
        raw_segments, result_img = interactive_infer_image_idino_m2m_auto(
            model,
            image_for_seg,
            prompt_level, all_classes, all_parts,
            thresh="0.0", text_size=TEXT_SIZE,
            hole_scale=100, island_scale=100, semantic=True,
        )


    # 2) 마스크 IOU 필터링을 위해 fg_mask 전달
    labeled_segments = util.postprocess_segments(
        raw_segments,
        image_for_seg,
        fg_mask           # ← NEW
    )
    
    # 5) 결과 이미지 저장 (선택적)
    filename = data.get("filename")
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds").replace(":", "-") + "Z"
    base_raw = filename.replace("/", "_").rsplit(".", 1)[0] if filename else uuid.uuid4().hex
    base_name = f"{ts}_{base_raw}"  # ← 시간 추가된 이름

    user = current_session["user"]
    version = current_session["version"]
    save_dir = os.path.join("logs", user, version, 'images', 'segmented_images', base_name)
    os.makedirs(save_dir, exist_ok=True)

    # (1) 결과 이미지 저장
    result_img.save(os.path.join(save_dir, "seg_result.png"))

    # (2) original 저장
    image_orig.save(os.path.join(save_dir, "original.png"))

    # (3) segments.json 저장
    with open(os.path.join(save_dir, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(labeled_segments, f, ensure_ascii=False, indent=2)

    # (4) 마스크 기반 이미지 저장
    seg_dir = os.path.join(save_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    
    seg_path_map = {}  # ★ seg_id -> 파일 경로
    
    for seg in labeled_segments:
        seg_id = seg["id"]
        label = seg.get("label", "unknown").replace(" ", "_")
        path  = seg.get("path")
        if not path:
            print(f"⚠️ {seg_id} has no path. Skipping.")
            continue

        mask_pil = util.path_to_mask(path, *image_orig.size).convert("L")
        original_rgba = image_orig.convert("RGBA")
        masked = Image.new("RGBA", original_rgba.size, (0, 0, 0, 0))
        masked.paste(original_rgba, mask=mask_pil)

        seg_fname = f"{seg_id}_label={label}_{base_name}.png" 
        seg_full_path = os.path.join(seg_dir, seg_fname)
        masked.save(seg_full_path)

        # 경로를 seg에 넣어두기 (CSV에서 바로 씀)
        seg["file_path"] = seg_full_path.replace("\\", "/")
        seg_path_map[seg_id] = seg["file_path"]
        
    # 5) 클라이언트 응답용 Base64
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    result_b64 = base64.b64encode(buf.getvalue()).decode()
    
    # 6) KG Snapshot 저장 ──────────────────────────────
    snap_csv = os.path.join("logs", user, version, "kg_snapshots.csv")
    write_header = not os.path.exists(snap_csv)

    with open(snap_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["ts", "subject", "predicate", "object", "seg_id", "label", "seg_path"])

        for seg in labeled_segments:
            seg_id   = seg.get("id")
            label    = seg.get("label")
            seg_path = seg.get("file_path", "")
            for t in seg.get("triples", []):
                writer.writerow([
                    ts,
                    seg_path,                         # subject = 실제 파일경로
                    t.get("predicate", ""),
                    t.get("object", ""),
                    seg_id,
                    label,
                    seg_path
                ])
    # 7) KG Total (중복 제거 + is-style-of 기준) ─────────────────
    total_csv = os.path.join("logs", user, version, "kg_total.csv")

    # 1) 기존 key 로드: (style, predicate, object)
    existing = set()
    if os.path.exists(total_csv):
        with open(total_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row["style"], row["predicate"], row["object"]))

    new_rows = [P]
    for seg in labeled_segments:
        seg_path = seg.get("file_path", "")
        seg_id   = seg.get("id")
        triples  = seg.get("triples", [])

        # style 트리플 & 속성 트리플 분리
        styles = [t["object"] for t in triples if t.get("predicate") == "is-style-of"]
        attrs  = [t for t in triples if t.get("predicate") not in {"is-style-of", "label", "name", "has-part"}]

        for style in styles:
            for a in attrs:
                key = (style, a.get("predicate",""), a.get("object",""))
                if key not in existing:
                    existing.add(key)
                    new_rows.append([
                        ts,
                        style,
                        a.get("predicate",""),
                        a.get("object",""),
                        seg_path,
                        seg_id
                    ])

    if new_rows:
        write_header = not os.path.exists(total_csv)
        with open(total_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ts", "style", "predicate", "object", "seg_path", "seg_id"])
            w.writerows(new_rows)

    # print(labeled_segments)
    return jsonify(
        segments           = labeled_segments,
        result_image_b64   = result_b64,
        # result_image_path=f"/logs/{user}/{version}/{base_name}/seg_result.png"
    )




_upload_cache: dict[tuple[int,str], str] = {}

def _upload_pil_as_file(pil_img: Image.Image, fname: str) -> str:
    """중복 업로드 방지 + 20 MB, 2048px 제한 대비."""
    key = (id(pil_img), fname)
    if key in _upload_cache:          # 이미 올렸으면 file_id 재사용
        return _upload_cache[key]

    # 👉 2048px 이하로 축소 + PNG 최적화
    w, h = pil_img.size
    if max(w, h) > 2048:
        scale = 2048 / max(w, h)
        pil_img = pil_img.resize((round(w*scale), round(h*scale)), Image.LANCZOS)

    buf = BytesIO()
    pil_img.save(buf, "PNG", optimize=True)
    buf.seek(0); buf.name = fname

    if buf.getbuffer().nbytes > 20*1024*1024:   # 20 MB 넘어가면 바로 에러
        raise RuntimeError(f"{fname} bigger than 20 MB after resize")

    file_id = client.files.create(file=buf, purpose="vision").id
    _upload_cache[key] = file_id
    return file_id

def _resize_if_needed(img: Image.Image, max_side=1024):
    w, h = img.size
    if max(w, h) <= max_side:
        return img           # 그대로
    scale = max_side / max(w, h)
    new_sz = (round(w*scale), round(h*scale))
    return img.resize(new_sz, Image.LANCZOS)

def _extract_image_b64(resp):
    """
    GPT-4o image_generation tool output → base64
    * 0.x (pre-release AssistantTools) : resp.output[..].result
    * 1.x (공식 SDK)                   : resp.tool_calls[..].response["image"]["base64"]
    * 단건 image_generation            : resp.image["base64"]
    """
    # ── 0.x ──────────────────────────────────────────
    if hasattr(resp, "output"):
        for o in resp.output:
            if getattr(o, "result", None):
                return o.result

    # ── 1.x  : tool_calls 경로 ───────────────────────
    if hasattr(resp, "tool_calls"):
        try:
            return resp.tool_calls[0].response["image"]["base64"]
        except (AttributeError, KeyError, IndexError):
            pass

    # ── 1.x  : 단건 image_generation 응답 ─────────────
    try:
        img = resp.image                    # dict?
        if isinstance(img, dict) and "base64" in img:
            return img["base64"]
    except AttributeError:
        pass

    # ── 못 찾으면 구조 찍고 에러 ──────────────────────
    import json, pprint
    dump_str = resp.model_dump_json(indent=2)  # 문자열로 변환!
    pprint.pprint(json.loads(dump_str))        # ← 슬라이스 ❌
    raise RuntimeError("image_generation 결과를 찾을 수 없습니다.")

# ───────── prompt에서 생성하기 !! ─────────

# @app.route("/apply_attributes", methods=["POST"])
# def apply_attributes():
#     try:
#         data = request.get_json(force=True) or {}
#         user    = (data.get("user")    or current_session["user"]).strip()
#         version = (data.get("version") or current_session["version"]).strip()
#         print("yes!")
#         # ───── [공통] Base 이미지 처리 ─────
#         base_b64 = data.get("base_image_b64")
#         if not base_b64:
#             return jsonify(error="Missing base_image_b64"), 400

#         base_img = Image.open(BytesIO(base64.b64decode(base_b64))).convert("RGB")
#         W, H = base_img.size

#         path_raw = data.get("target_path")
#         if not path_raw:
#             return jsonify(error="Missing target_path"), 400

#         if isinstance(path_raw, (list, tuple)):
#             path_raw = " ".join(path_raw)

#         # 1) 마스크 생성
#         seg_mask_L = util.path_to_mask(path_raw, W, H)
#         mask_rgba = Image.new("RGBA", (W, H), (0, 0, 0, 255))
#         ImageDraw.Draw(mask_rgba).bitmap((0, 0), seg_mask_L, fill=(0, 0, 0, 0))

#         # 2) 파일 업로드
#         base_id = _upload_pil_as_file(_resize_if_needed(base_img), "base.png")
#         mask_id = _upload_pil_as_file(_resize_if_needed(mask_rgba), "mask.png")

#         # ───── [분기: edit or transfer] ─────
#         mode = data.get("mode", "transfer")
#         print(mode)
#         prompt = ""
#         ref_id = None

#         if mode == "transfer":
#             src_seg = data.get("source_segment")
#             if not src_seg:
#                 return jsonify(error="Missing source_segment"), 400

#             src_triples = src_seg.get("triples", [])
#             prompt = util.triples_to_prompt(src_triples) or "apply the reference style"

#             bbox = src_seg.get("bbox")
#             if not bbox:
#                 return jsonify(error="Missing bbox in source_segment"), 400

#             ref_img = base_img.crop((
#                 int(bbox["x"] - bbox["width"] / 2),
#                 int(bbox["y"] - bbox["height"] / 2),
#                 int(bbox["x"] + bbox["width"] / 2),
#                 int(bbox["y"] + bbox["height"] / 2),
#             ))
#             ref_id = _upload_pil_as_file(_resize_if_needed(ref_img), "ref.png")

#         elif mode == "edit":
#             tgt_seg = data.get("target_segment", {})
#             new_triples = data.get("new_triples", [])
#             old_triples = tgt_seg.get("triples", [])
#             print(tgt_seg, "old_triples:", old_triples, "new_triples:", new_triples)
#             prompt = util.diff_triples_to_prompt(old_triples, new_triples) or "edit the segment attributes"

#         else:
#             return jsonify(error=f"Invalid mode: {mode}"), 400

#         print("🧠 Final prompt:", prompt)

#         # ───── GPT 호출 ─────
#         user_inputs = [
#             {"type": "input_text", "text": prompt},
#             {"type": "input_image", "file_id": base_id}
#         ]
#         if ref_id:
#             user_inputs.append({"type": "input_image", "file_id": ref_id})

#         tool_cfg = {
#             "type": "image_generation",
#             "quality": "high",
#             "input_image_mask": {"file_id": mask_id}
#         }


#         resp = client.responses.create(
#             model="gpt-4o",
#             input=[{"role": "user", "content": user_inputs}],
#             tools=[tool_cfg],
#             tool_choice={"type": "image_generation"}
#         )

#         # 4) 결과 처리
#         gen_b64 = _extract_image_b64(resp)
#         gen_img = Image.open(BytesIO(base64.b64decode(gen_b64))).convert("RGB")

#         fname = f"full_{uuid.uuid4().hex}.png"
#         fpath = os.path.join(GENERATED_DIR, fname)
#         gen_img.save(fpath)
#         overlay = base_img.copy().convert("RGBA")
#         red_mask = Image.new("RGBA", mask_rgba.size, (255, 0, 0, 120))  # 붉은 반투명 오버레이
#         overlay.paste(red_mask, (0, 0), mask_rgba)  # 마스크를 알파채널로 붙임

#         debug_overlay_path = os.path.join("debug", f"overlay_{uuid.uuid4().hex}.png")
#         os.makedirs(os.path.dirname(debug_overlay_path), exist_ok=True)
#         overlay.save(debug_overlay_path)
#         print(f"[DEBUG] Saved overlay image to {debug_overlay_path}")
#         with open(fpath, "rb") as f:
#             full_b64 = base64.b64encode(f.read()).decode()

#         raw_segs, _ = interactive_infer_image_idino_m2m_auto(
#             model, gen_img, prompt_level,
#             all_classes, all_parts,
#             thresh="0.0", text_size=TEXT_SIZE,
#             hole_scale=100, island_scale=100, semantic=True,
#         )
#         full_segments = util.postprocess_segments(raw_segs, gen_img, None)
@app.route("/apply_attributes", methods=["POST"])
def apply_attributes():
    try:
        data = request.get_json(force=True) or {}
        user    = (data.get("user")    or current_session["user"]).strip()
        version = (data.get("version") or current_session["version"]).strip()

        # ───── Base 이미지 처리 ─────
        base_b64 = data.get("base_image_b64")
        if not base_b64:
            return jsonify(error="Missing base_image_b64"), 400
        base_img = Image.open(BytesIO(base64.b64decode(base_b64))).convert("RGB")
        W, H = base_img.size

        # ───── 마스크 생성 ─────
        path_raw = data.get("target_path")
        if not path_raw:
            return jsonify(error="Missing target_path"), 400
        if isinstance(path_raw, (list, tuple)):
            path_raw = " ".join(path_raw)
        seg_mask_L = util.path_to_mask(path_raw, W, H)
        mask_rgba = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        ImageDraw.Draw(mask_rgba).bitmap((0, 0), seg_mask_L, fill=(0, 0, 0, 0))

        # ───── Prompt 생성 ─────
        mode = data.get("mode", "transfer")
        prompt = ""
        if mode == "transfer":
            src_seg = data.get("source_segment", {})
            src_triples = src_seg.get("triples", [])
            prompt = util.triples_to_prompt(src_triples) or "apply the reference style"
        elif mode == "edit":
            tgt_seg = data.get("target_segment", {})
            new_triples = data.get("new_triples", [])
            old_triples = tgt_seg.get("triples", [])
            prompt = util.diff_triples_to_prompt(old_triples, new_triples) or "edit the segment attributes"
        else:
            return jsonify(error=f"Invalid mode: {mode}"), 400

        print("🧠 Final prompt:", prompt)

        # ───── OpenAI 이미지 편집 호출 ─────
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_base, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_mask:
            base_img.save(f_base.name)
            mask_rgba.save(f_mask.name)
            debug_dir = os.path.join("logs", user, version, "input_to_gpt")
            os.makedirs(debug_dir, exist_ok=True)
            debug_base_path = os.path.join(debug_dir, "base_input.png")
            debug_mask_path = os.path.join(debug_dir, "mask_input.png")
            base_img.save(debug_base_path)
            mask_rgba.save(debug_mask_path)
            
            result = client.images.edit(
                model="gpt-image-1",
                image=open(f_base.name, "rb"),
                mask=open(f_mask.name, "rb"),
                prompt=prompt,
                size="1024x1024",  # 고정 사이즈
                n=1
            )

        image_base64 = result.data[0].b64_json
        gen_img = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")

        # 결과 저장
        fname = f"full_{uuid.uuid4().hex}.png"
        fpath = os.path.join(GENERATED_DIR, fname)
        gen_img.save(fpath)

        with open(fpath, "rb") as f:
            full_b64 = base64.b64encode(f.read()).decode()

        # 1) 배경 제거
        fg_rgba, fg_mask = util.remove_background(gen_img, smooth_edges=False)

        # 2) 흰 배경에 전경만 합성해서 세그멘테이션용 이미지 생성
        canvas = Image.new("RGB", fg_rgba.size, (255, 255, 255))
        canvas.paste(fg_rgba, mask=fg_rgba.split()[-1])
        image_for_seg = canvas

        # 3) Semantic-SAM 세그멘테이션은 removebg 결과(image_for_seg) 기준
        raw_segs, _ = interactive_infer_image_idino_m2m_auto(
            model,
            image_for_seg,
            prompt_level,
            all_classes, all_parts,
            thresh="0.0", text_size=TEXT_SIZE,
            hole_scale=100, island_scale=100, semantic=True,
        )
        full_segments = util.postprocess_segments(raw_segs, image_for_seg, fg_mask)
        
        # 저장
        ts = datetime.datetime.utcnow().isoformat(timespec="seconds").replace(":", "-") + "Z"
        user = current_session["user"]
        version = current_session["version"]
        save_root = os.path.join("logs", user, version, 'images',"generated_images", f"{ts}_apply")

        os.makedirs(save_root, exist_ok=True)
        gen_img.save(os.path.join(save_root, "full.png"))
        base_img.save(os.path.join(save_root, "original.png"))
        
        with open(os.path.join(save_root, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(full_segments, f, ensure_ascii=False, indent=2)
        seg_dir = os.path.join(save_root, "segments")
        os.makedirs(seg_dir, exist_ok=True)

        for seg in full_segments:
            seg_id = seg["id"]
            label = seg.get("label", "unknown").replace(" ", "_")
            path = seg.get("path")
            if not path:
                continue
            mask_pil = util.path_to_mask(path, *gen_img.size).convert("L")
            rgba_img = gen_img.convert("RGBA")
            masked = Image.new("RGBA", rgba_img.size, (0, 0, 0, 0))
            masked.paste(rgba_img, mask=mask_pil)
            seg_fname = f"{seg_id}_label={label}_{ts}.png"                      # ★ ADD
            seg_full_path = os.path.join(seg_dir, seg_fname)                 # ★ ADD
            masked.save(seg_full_path)                                       # ★ CHG
            seg["file_path"] = seg_full_path.replace("\\", "/")              # ★ ADD
            
        # 클라이언트에서 접근할 수 있는 URL 생성
        host = request.host.split(":")[0]
        url = f"http://{host}:5010/generated/{fname}"
        # ---------- ★★★ ADD: KG Snapshot & Total CSV 기록 ★★★ ----------
        snap_csv = os.path.join("logs", user, version, "kg_snapshots.csv")
        write_header = not os.path.exists(snap_csv)
        with open(snap_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ts", "subject", "predicate", "object", "seg_id", "label", "seg_path"])
            for seg in full_segments:
                seg_id   = seg.get("id")
                label    = seg.get("label")
                seg_path = seg.get("file_path", "")
                for t in seg.get("triples", []):
                    w.writerow([
                        ts,
                        seg_path,                       # subject
                        t.get("predicate", ""),
                        t.get("object", ""),
                        seg_id,
                        label,
                        seg_path
                    ])

        # --- KG Total (is-style-of × attr 조합) ---
        total_csv = os.path.join("logs", user, version, "kg_total.csv")
        existing = set()
        if os.path.exists(total_csv):
            with open(total_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.add((row["style"], row["predicate"], row["object"]))

        new_rows = []
        for seg in full_segments:
            seg_path = seg.get("file_path", "")
            seg_id   = seg.get("id")
            triples  = seg.get("triples", [])
            styles = [t["object"] for t in triples if t.get("predicate") == "is-style-of"]
            attrs  = [t for t in triples if t.get("predicate") not in {"is-style-of", "label", "name", "has-part"}]
            for style in styles:
                for a in attrs:
                    key = (style, a.get("predicate",""), a.get("object",""))
                    if key not in existing:
                        existing.add(key)
                        new_rows.append([ts, style, a.get("predicate",""), a.get("object",""), seg_path, seg_id])

        if new_rows:
            write_header = not os.path.exists(total_csv)
            with open(total_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["ts", "style", "predicate", "object", "seg_path", "seg_id"])
                w.writerows(new_rows)
# ---------- ★★★ END ADD ★★★ ----------

        return jsonify({
            "full_image_path": url,
            "full_image_b64": full_b64,
            "full_segments": full_segments
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 500


# ────────────────────────────────────────────────────────────────────
@app.route("/generate_image_from_attributes", methods=["POST"])
def generate_image_from_attributes():
    """
    • prompt:   "modern table" …
    • triples:  [{subject,predicate,object}, …]  ─ (옵션) 직접 지정 속성
    • graph:    전체 KG ─ triples 없을 때 **샘플링 기반 Bayesian** 추론용
    • top_k / alpha / temperature: (옵션) sampling 파라미터
    """
    try:
        # ── 0) 요청 파싱 ──────────────────────────────────────────
        data = request.get_json(force=True) or {}
        user    = (data.get("user")    or current_session["user"]).strip()
        version = (data.get("version") or current_session["version"]).strip()
        IS_BASELINE = version.lower().startswith("baseline") 
        prompt  = data.get("prompt", "").strip()
        triples = data.get("triples", []) or []
        graph   = data.get("graph", [])              # optional

        # optional sampling params (전송 안 하면 기본값 사용)
        top_k        = data.get("top_k",        15)
        alpha        = data.get("alpha",        0.5)
        temperature  = data.get("temperature",  1.0)

        if not prompt:
            return jsonify(error="Missing prompt"), 400

        # ── 1) fallback: Bayesian 추론 ─────────────────────────────
        if not triples and graph:
                triples = util.infer_triples_from_hierarchical_model_sampling(
                prompt,
                graph,
                top_k=top_k,
                alpha=alpha,
                temperature=temperature
            ) or []

        # ── 2) 객체 정체성 트리플(is‑a/label/name) 제거 ────────────
        attr_triples = [
            tri for tri in triples
            if tri.get("predicate") not in {"is-a", "label", "name", "has-part"}
        ]
        attr_prompt = util.triples_to_prompt(attr_triples)      # “walnut top, chrome legs”
        prompt_text = f"{prompt}, {attr_prompt}" if attr_prompt else prompt

        # ── 3) prompt에서 객체명 추출 → system 메시지에 강제 ────────
        def _extract_object_name(text: str) -> str:
            """가장 마지막 토큰을 객체명으로 간주 (단순 버전)"""
            tokens = re.sub(r"[^\w\s-]", " ", text.lower()).split()
            return tokens[-1] if tokens else "object"

        object_name = _extract_object_name(prompt)

        print(f"[Prompt]   {prompt_text}")
        print(f"[Object]   {object_name}")
        print(f"[Triples]  {attr_triples}")

        # ── 4) GPT‑4o 이미지 생성 ──────────────────────────────────
        resp = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "system",
                "content": (
                    "You are a product image generator. "
                    f"The main subject MUST be a {object_name}. "
                    f"Generate **a single {object_name}** only, centered, "
                    "with no additional objects or scenery. "
                    # "Use a plain or white background. "/
                    "Return no text—only the image_generation tool."
                    "Ignore any conflicting context. "
                )
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt_text}]
                }
            ],
            tools=[{"type": "image_generation", "quality": "high"}]
        )
        gen_b64 = _extract_image_b64(resp)
        if not gen_b64:
            return jsonify(error="no image generated"), 500

        # # ── 5) 생성 이미지 파일 저장 ───────────────────────────────
        # full      = Image.open(BytesIO(base64.b64decode(gen_b64))).convert("RGB")
        # full_name = f"gen_{uuid.uuid4().hex}.png"
        # full_path = os.path.join(GENERATED_DIR, full_name)
        # full.save(full_path, "PNG")

        # # ── 6) 배경 제거 + 세그멘테이션 ────────────────────────────
        # if IS_BASELINE:
        #     with open(full_path, "rb") as f:
        #         full_b64 = base64.b64encode(f.read()).decode()
        #     full_url = f"http://localhost:5010/generated/{full_name}"
        #     return jsonify(
        #         full_image_path=full_url,
        #         full_image_b64=full_b64,
        #         full_segments=[]   # ← 프론트가 기대하는 키 유지
        #     )
        # fg_rgba, fg_mask = util.remove_background(full, smooth_edges=False)
        # canvas = Image.new("RGB", fg_rgba.size, (255, 255, 255))
        # canvas.paste(fg_rgba, mask=fg_rgba.split()[-1])
        # image_for_seg = canvas

        # with torch.no_grad(), AMP_CTX:
        #     raw_segs, _ = interactive_infer_image_idino_m2m_auto(
        #         model,
        #         image_for_seg,
        #         prompt_level,
        #         all_classes, all_parts,
        #         thresh="0.0",
        #         text_size=TEXT_SIZE,
        #         hole_scale=100,
        #         island_scale=100,
        #         semantic=True,
        #     )
        # full_segments = util.postprocess_segments(raw_segs, image_for_seg, fg_mask)
        
        
        
        
                # ── 5) 생성 이미지 파일 저장 ───────────────────────────────
        full      = Image.open(BytesIO(base64.b64decode(gen_b64))).convert("RGB")
        full_name = f"gen_{uuid.uuid4().hex}.png"
        full_path = os.path.join(GENERATED_DIR, full_name)
        full.save(full_path, "PNG")

        # ── 6) 세그멘테이션만 수행 (배경 제거 스킵) ─────────────────
        if IS_BASELINE:
            with open(full_path, "rb") as f:
                full_b64 = base64.b64encode(f.read()).decode()
            full_url = f"http://localhost:5010/generated/{full_name}"
            return jsonify(
                full_image_path=full_url,
                full_image_b64=full_b64,
                full_segments=[]
            )

        # ★ 배경 제거 안 하고 원본 이미지를 그대로 세그멘테이션에 사용
        image_for_seg = full

        with torch.no_grad(), AMP_CTX:
            raw_segs, _ = interactive_infer_image_idino_m2m_auto(
                model,
                image_for_seg,
                prompt_level,
                all_classes, all_parts,
                thresh="0.0",
                text_size=TEXT_SIZE,
                hole_scale=100,
                island_scale=100,
                semantic=True,
            )

        # ★ 전체가 foreground라고 가정하는 더미 마스크 생성
        import numpy as np
        fg_mask = np.ones((image_for_seg.height, image_for_seg.width), dtype=np.uint8) * 255

        # 여기서 기존 호출 유지
        full_segments = util.postprocess_segments(raw_segs, image_for_seg, fg_mask)
        
        
        
        # 6.1) 세그먼트 결과 저장
        ts = datetime.datetime.utcnow().isoformat(timespec="seconds").replace(":", "-") + "Z"

        save_root = os.path.join("logs", user, version, 'images', "generated_images", f"{ts}_from_prompt")

        os.makedirs(save_root, exist_ok=True)

        # 생성 이미지 저장
        full.save(os.path.join(save_root, "full.png"))

        # (선택) 복사본 저장 → 필요 시 "original"도 full로 대체 가능
        # full.copy().save(os.path.join(save_root, "original.png"))

        # segments.json 저장
        with open(os.path.join(save_root, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(full_segments, f, ensure_ascii=False, indent=2)

        # segment 마스크 이미지 저장
        seg_dir = os.path.join(save_root, "segments")
        os.makedirs(seg_dir, exist_ok=True)

        for seg in full_segments:
            seg_id = seg["id"]
            label = seg.get("label", "unknown").replace(" ", "_")
            path = seg.get("path")
            if not path:
                continue
            mask_pil = util.path_to_mask(path, *full.size).convert("L")
            rgba_img = full.convert("RGBA")
            masked = Image.new("RGBA", rgba_img.size, (0, 0, 0, 0))
            masked.paste(rgba_img, mask=mask_pil)
            seg_fname = f"{seg_id}_label={label}.png"                       # ★ ADD
            seg_full_path = os.path.join(seg_dir, seg_fname)                 # ★ ADD
            masked.save(seg_full_path)                                       # ★ CHG
            seg["file_path"] = seg_full_path.replace("\\", "/")  
                        
        # ── 7) 응답 반환 ───────────────────────────────────────────
        with open(full_path, "rb") as f:
            full_b64 = base64.b64encode(f.read()).decode()
        full_url = f"http://localhost:5010/generated/{full_name}"

        # ---------- ★★★ ADD: KG Snapshot & Total CSV 기록 ★★★ ----------
        snap_csv = os.path.join("logs", user, version, "kg_snapshots.csv")
        write_header = not os.path.exists(snap_csv)
        with open(snap_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ts", "subject", "predicate", "object", "seg_id", "label", "seg_path"])
            for seg in full_segments:
                seg_id   = seg.get("id")
                label    = seg.get("label")
                seg_path = seg.get("file_path", "")
                for t in seg.get("triples", []):
                    w.writerow([
                        ts,
                        seg_path,                       # subject
                        t.get("predicate", ""),
                        t.get("object", ""),
                        seg_id,
                        label,
                        seg_path
                    ])

        # --- KG Total (is-style-of × attr 조합) ---
        total_csv = os.path.join("logs", user, version, "kg_total.csv")
        existing = set()
        if os.path.exists(total_csv):
            with open(total_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing.add((row["style"], row["predicate"], row["object"]))

        new_rows = []
        for seg in full_segments:
            seg_path = seg.get("file_path", "")
            seg_id   = seg.get("id")
            triples  = seg.get("triples", [])
            styles = [t["object"] for t in triples if t.get("predicate") == "is-style-of"]
            attrs  = [t for t in triples if t.get("predicate") not in {"is-style-of", "label", "name", "has-part"}]
            for style in styles:
                for a in attrs:
                    key = (style, a.get("predicate",""), a.get("object",""))
                    if key not in existing:
                        existing.add(key)
                        new_rows.append([ts, style, a.get("predicate",""), a.get("object",""), seg_path, seg_id])

        if new_rows:
            write_header = not os.path.exists(total_csv)
            with open(total_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["ts", "style", "predicate", "object", "seg_path", "seg_id"])
                w.writerows(new_rows)
        # ---------- ★★★ END ADD ★★★ ----------


        return jsonify(
            full_image_path=full_url,
            full_image_b64=full_b64,
            full_segments=full_segments
        )

    except Exception as e:
        traceback.print_exc()
        if getattr(e, "http_status", None):
            print("◆ OpenAI status:", e.http_status, file=sys.stderr)
        if getattr(e, "error", None):
            print("◆ OpenAI payload:", e.error, file=sys.stderr)
        return jsonify(error=str(e)), 500
# =====================================================================

@app.post("/search_clip")
def search_clip():
    # 0) 쿼리 파라미터
    data = request.get_json(silent=True) or {}
    text_query = data.get("query", "").strip()
    if not text_query:
        return jsonify([])

    # 1) 텍스트 → CLIP 임베딩
    inputs = clip_proc(text=[text_query], return_tensors="pt")
    with torch.no_grad():
        inputs   = {k: v.to(DEVICE) for k, v in inputs.items()}   # ← GPU로
        txt_feat = clip_model.get_text_features(**inputs)
    k = faiss_index.ntotal 
    # 2) FAISS 검색
    D, I = faiss_index.search(
        txt_feat.cpu().numpy().astype("float32"),   # GPU → CPU
        k = faiss_index.ntotal 
    )
    results = [image_paths[idx] for idx in I[0]]
    return jsonify(results)

@app.post("/init_session")
def init_session():
    global current_session
    data    = request.get_json(silent=True) or {}
    user    = (data.get("user") or "anonymous").strip()
    version = (data.get("version") or "").strip()
    if not version:
        version = datetime.datetime.utcnow().isoformat(timespec="seconds").replace(":", "-") + "Z"

    # ★ 여기서 꼭 업데이트!
    current_session.update({"user": user, "version": version})

    # 이하 기존 코드 동일 …
    sess = SESSION_ROOT / user / version
    sess.mkdir(parents=True, exist_ok=True)
    util.append_action(sess, {"ts": util.now(), "action": "init_session"})

    # 3) CSV에도 한 줄 추가 ─ actions_images.csv 쪽
    csv1 = sess / "actions_images.csv"
    write_header = not csv1.exists()
    with open(csv1, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(["ts","action","before_image","after_image","segment","prompt","triple"])
        wr.writerow([util.now(),"init_session","","","","",""])

    return jsonify(status="ok", path=str(sess))

@app.post("/log_action")
def log_action():
    """
    - before/after 이미지만 images/ 아래 저장
    - 세그먼트 PNG( segment / triple / segments_b64 )는 모두 kg/<ts_action>/segs/ 아래 저장
    - kg_current / kg_delta / snapshots 는 kg/<ts_action>/ 에 JSON 저장
    - current_style.json 은
        1) kg_current.style 가 있으면 그걸,
        2) 없으면 snapshots.style 을,
        3) 둘 다 없으면 생략
      해서 반드시 시도.
    - actions_images.csv 는 기존 유지
    """
    import json, csv, os, traceback, copy
    from pathlib import Path
    from typing import Union

    data = request.get_json(force=True)

    # 1) 필수값
    user       = data.get("user")
    ver        = data.get("version")
    action_raw = data.get("action")
    if not user or not ver or not action_raw:
        return jsonify(error="user, version, action required"), 400
    IS_BASELINE = (ver or "").lower() == "baseline"

    # 2) 세션 & ts
    sess = util.session_dir(user, ver)              # logs/<user>/<ver>
    ts   = util.now()                                # "2025-07-28T06:47:09Z"
    kg_key = f"{ts}_{action_raw}".replace(":", "-").replace(".", "-")

    # ───────── helpers ─────────
    def _dump_json(path: Path, obj, out_paths: dict, key: str):
        if obj is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")
        out_paths[key] = str(path)

    def _inject_seg_paths(obj, seg_paths):
        if isinstance(obj, dict):
            sid = obj.get("id")
            if sid in seg_paths:
                obj["img_path"] = seg_paths[sid]
            for v in obj.values():
                _inject_seg_paths(v, seg_paths)
        elif isinstance(obj, list):
            for v in obj:
                _inject_seg_paths(v, seg_paths)

    def _save_png_rel(sess_path: Path, rel_path: Union[str, Path], b64: str) -> str:
        dst = sess_path / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        # data URI 제거 + 패딩
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        b64 += "=" * (-len(b64) % 4)
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGBA")
        img.save(dst)
        return dst.relative_to(sess_path).as_posix()

    # 3) before / after -> images/
    (sess / "images").mkdir(parents=True, exist_ok=True)
    img_prefix_rel = f"images/{ts}_{action_raw}"
    saved_before = saved_after = None

    before_img = data.get("before_image") or {}
    after_img  = data.get("after_image")  or {}

    if before_img.get("b64"):
        saved_before = util.save_image_file(sess, f"{img_prefix_rel}_before.png", b64=before_img["b64"])
    if after_img.get("b64"):
        saved_after  = util.save_image_file(sess, f"{img_prefix_rel}_after.png",  b64=after_img["b64"])

    # 4) KG 디렉토리
    kg_base   = sess / "kg" / kg_key
    kg_segdir = kg_base / "segs"
    kg_segdir.mkdir(parents=True, exist_ok=True)

    # 5) 단일 segment / triple b64 -> kg/segs/
    saved_seg = None
    seg = data.get("segment")
    if seg and seg.get("b64"):
        seg_b64 = seg["b64"]
        if seg.get("path"):
            seg_b64 = util.mask_by_svgpath(seg_b64, seg["path"])
        seg_id_safe = Path(str(seg.get("id", "no_seg"))).name
        rel = Path("kg") / kg_key / "segs" / f"main_{seg_id_safe}.png"
        saved_seg = _save_png_rel(sess, rel, seg_b64)

    triples_in = data.get("triple") or []
    if not isinstance(triples_in, list):
        triples_in = [triples_in]

    triples_out = []
    for tri_raw in triples_in:
        if isinstance(tri_raw, dict):
            tri = tri_raw.copy()
        elif isinstance(tri_raw, (list, tuple)) and len(tri_raw) == 2:
            tri = {"predicate": tri_raw[0], "object": tri_raw[1]}
        else:
            continue

        b64 = tri.pop("b64", None)
        if b64:
            seg_id_safe = Path(str(tri.get("segId", "no_seg"))).name
            role_safe   = Path(str(tri.get("role", "no_role"))).name
            rel = Path("kg") / kg_key / "segs" / f"{role_safe}_{seg_id_safe}.png"
            saved_fname = _save_png_rel(sess, rel, b64)
            tri["img"]  = saved_fname
        triples_out.append(tri)

    # 6) KG JSON 저장
    kg_current_all = data.get("kg_current")   or {}
    kg_delta_all   = data.get("kg_delta")     or {}
    seg_b64_map    = data.get("segments_b64") or {}
    snapshots_all  = data.get("snapshots")    or {}
    # ⬇️ 저장본 전용(second bundle)용 필드 (옵션)
    kg_current_saved   = data.get("kg_current_saved")   or {}
    kg_delta_saved     = data.get("kg_delta_saved")     or {}
    seg_b64_map_saved  = data.get("segments_b64_saved") or {}
    snapshots_saved    = data.get("snapshots_saved")    or {}
    should_save_kg = any([kg_current_all, kg_delta_all, seg_b64_map, snapshots_all]) 
       
    def save_kg_bundle(sess_path: Path, kg_key: str,
                    kg_cur: dict = None,
                    kg_del: dict = None,
                    seg_map: dict = None,
                    snaps:  dict = None):
        import copy
        # ─── 저장할 내용이 없으면 스킵 ───
        has_segs    = bool(seg_map)
        has_graph   = bool(kg_cur and (kg_cur.get("graph") or kg_cur.get("triples")))
        # current_full.json 은 kg_cur 전체로 저장하므로 kg_cur이 빈 dict면 스킵
        if not has_segs and not has_graph:
            return {}

        # ✅ 반드시 먼저 선언
        base_dir = sess_path / "kg" / kg_key
        seg_dir  = base_dir / "segs"
        seg_dir.mkdir(parents=True, exist_ok=True)

        out_paths: dict = {}
        seg_paths: dict = {}

        # (1) segments_b64 저장
        if seg_map:
            for seg_id, b64 in seg_map.items():
                sid = str(seg_id).replace("/", "_").replace("\\", "_")
                rel = Path("kg") / kg_key / "segs" / f"{sid}__{kg_key}.png"
                seg_paths[seg_id] = _save_png_rel(sess_path, rel, b64)
            out_paths["seg_paths"] = seg_paths

        def inject(obj):
            _inject_seg_paths(obj, seg_paths)

        # --- STYLE 저장 로직 단순화 ---
        style_obj = None
        if kg_cur and isinstance(kg_cur, dict):
            style_obj = kg_cur.get("style") or kg_cur.get("image", {}).get("style")
        if style_obj is None and snaps and isinstance(snaps, dict):
            style_obj = snaps.get("style")

        # (2) current
        if kg_cur:
            cur_p = copy.deepcopy(kg_cur)
            inject(cur_p)

            if "graph" in cur_p or "triples" in cur_p:
                _dump_json(base_dir / "current_graph.json",   cur_p.get("graph"),   out_paths, "current_graph")
                _dump_json(base_dir / "current_triples.json", cur_p.get("triples"), out_paths, "current_triples")
                _dump_json(base_dir / "current_full.json",    cur_p,                out_paths, "current_full")

            if "image" in cur_p:
                _dump_json(base_dir / "current_image.json", cur_p["image"], out_paths, "current_image")


        # --- 스타일 요약 저장 ---
        # triples 우선 사용, 없으면 snapshots의 triples 사용(선택)
        triples_src = (kg_cur or {}).get("triples") or (snaps or {}).get("triples") or []
        style_summary = util.summarize_styles(triples_src, seg_paths)
        # print("style-graph generated:", style_summary)
        if style_summary:
            _dump_json(base_dir / "current_style_summary.json",
                        style_summary, out_paths, "current_style_summary")

        # if style_obj is not None:
        #     style_copy = copy.deepcopy(style_obj)
        #     inject(style_copy)
        #     _dump_json(base_dir / "current_style.json", style_copy, out_paths, "current_style")
        #     print("[DEBUG] current_style saved")

        # (3) delta
        if kg_del:
            del_p = copy.deepcopy(kg_del)
            inject(del_p)

            if "graph" in del_p or "triples" in del_p:
                _dump_json(base_dir / "delta_graph.json",   del_p.get("graph"),   out_paths, "delta_graph")
                _dump_json(base_dir / "delta_triples.json", del_p.get("triples"), out_paths, "delta_triples")
                _dump_json(base_dir / "delta_full.json",    del_p,                out_paths, "delta_full")

            if "style" in del_p:
                _dump_json(base_dir / "delta_style.json", del_p["style"], out_paths, "delta_style")
            if "image" in del_p:
                _dump_json(base_dir / "delta_image.json", del_p["image"], out_paths, "delta_image")

        # (4) snapshots
        if snaps:
            snaps_p = copy.deepcopy(snaps)
            inject(snaps_p)
            _dump_json(base_dir / "snapshots_raw.json", snaps_p, out_paths, "snapshots_raw")
            if "image_full" in snaps_p:
                _dump_json(base_dir / "snapshots_image_full.json", snaps_p["image_full"], out_paths, "snapshots_image_full")
            if "style" in snaps_p:
                _dump_json(base_dir / "snapshots_style.json", snaps_p["style"], out_paths, "snapshots_style")

        return out_paths

    kg_files = {}
    kg_files_saved = {}
    if not IS_BASELINE and should_save_kg:
        try:
            kg_files = save_kg_bundle(
                sess, kg_key,
                kg_cur=kg_current_all,
                kg_del=kg_delta_all,
                seg_map=seg_b64_map,
                snaps=snapshots_all
            )

            # ▶ 저장본 번들이 따로 왔으면 한 번 더 저장
            if kg_current_saved or kg_delta_saved or seg_b64_map_saved or snapshots_saved:
                kg_key_saved = f"{kg_key}_saved"
                kg_files_saved = save_kg_bundle(
                    sess, kg_key_saved,
                    kg_cur=kg_current_saved,
                    kg_del=kg_delta_saved,
                    seg_map=seg_b64_map_saved,
                    snaps=snapshots_saved
                )
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=str(e)), 500
    # 7) 액션 JSON 로그
    record = {
        "ts": ts,
        "action": action_raw,
        "before_image": saved_before,
        "after_image":  saved_after,
        "segment":      saved_seg,
        "prompt":       data.get("prompt"),
        "kg_current":   kg_current_all,
        "kg_delta":     kg_delta_all,
        "triple":       triples_out,
        "kg_files":     kg_files,
        "kg_files_saved": kg_files_saved,
        "kg_key":       kg_key
    }
    util.append_action(sess, record)
    
    SKIP_CSV = {"kg_auto", "kg_auto_saved"}
    if action_raw not in SKIP_CSV:
        csv1_path = os.path.join(sess, "actions_images.csv")
        write_header1 = not os.path.exists(csv1_path)
        with open(csv1_path, "a", newline="", encoding="utf-8") as f1:
            writer1 = csv.writer(f1)
            if write_header1:
                writer1.writerow([
                    "ts", "action",
                    "before_image", "after_image",
                    "segment", "prompt", "triple_json"
                ])
            writer1.writerow([
                ts, action_raw,
                saved_before, saved_after,
                saved_seg, data.get("prompt"),
                json.dumps(triples_out, ensure_ascii=False)
            ])

    return jsonify(ok=True, kg_files=kg_files)


if __name__ == "__main__":
    # 큰 이미지를 올리는 경우를 대비해 업로드 한도 ↑
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB
    app.run(host="0.0.0.0", port=5010, debug=True, use_reloader=False)
