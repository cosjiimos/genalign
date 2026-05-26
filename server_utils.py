from __future__ import annotations
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageChops
import torch, os, re, base64, uuid
from io import BytesIO
from math import ceil
import openai, base64, io, os
import json, datetime, shutil, base64, binascii       
from typing import List, Dict, Optional, Any
from svgpathtools import parse_path 
from pathlib import Path
from typing import Tuple , Any,Union, Literal
import cv2 , numpy as np
from rembg import remove, new_session
from collections import defaultdict, Counter
import re, random, numpy as np
import traceback    
# ───────── Semantic-SAM 의존 코드 ─────────
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from utils.arguments import load_opt_from_config_file
from tasks import interactive_infer_image_idino_m2m_auto
from utils.constants import COCO_PANOPTIC_CLASSES
# ──────────────────────────────────────────

IGNORED_LABELS = {"background", "shadow", "floor", "wall", "ceiling"}
# -----------------------------------------------------------



# ─────────────────────────────────── Segment Scale, Position 맞추기  ─────────────────────────────────── 

def scale_and_offset_path(path_d, sx, sy, pad_x, pad_y):
    """SVG path 좌표(문자열) → 패딩 제거 + 원본 해상도 스케일"""
    return re.sub(
        r"([0-9.]+)[ ,]([0-9.]+)",
        lambda m: f"{(float(m.group(1))-pad_x)*sx},{(float(m.group(2))-pad_y)*sy}",
        path_d,
    )

def split_obj_part(label: str):
    """'sofa-armrest' → ('sofa', 'armrest')"""
    return label.split("-", 1) if "-" in label else (label, None)

def ensure_closed_path(path_d: str) -> str:
    """path 끝이 Z(닫기)로 안 끝날 경우 추가"""
    return path_d.strip() + (" Z" if not path_d.strip().lower().endswith("z") else "")


def postprocess_segments(
    raw_segments: List[Dict],
    orig_img:    Image.Image,
    fg_mask:     Image.Image,
) -> List[Dict]:
    """
    - raw_segments : SAM에서 뽑은 path 리스트
    - orig_img     : white-bg composite(RGB)
    - fg_mask      : remove_background()가 반환한 L-mode 마스크
    """
    out = []
    fg_np = (np.array(fg_mask) > 0) if fg_mask is not None else None
    global_styles = detect_image_style(orig_img)
    
    for seg in raw_segments:
        # 1) path 보정 (닫기 명령 추가)
        path_closed = ensure_closed_path(seg["path"])

        # 2) path → 마스크 (L-mode) → numpy
        seg_mask_pil = path_to_mask(path_closed, *orig_img.size)
        seg_np       = (np.array(seg_mask_pil) > 0)

        # 3) IOU + 면적 필터링
        if fg_np is not None:
            inter = np.logical_and(seg_np, fg_np).sum()
            area  = seg_np.sum()
            if area == 0 or area < 20 or (inter / area) < IOU_THRESHOLD:
                continue

        # 4) 유효한 전경 segment 처리
        seg_img = mask_segment(orig_img, path_closed)
        label   = gpt_label_segment(seg_img, orig_img, path_closed)
        if label.lower() in IGNORED_LABELS:
            continue

        obj, part   = (label.split("-",1) + [None])[:2]
        seg_id      = f"seg_{seg['id']}"

        print("build triples")
        # 5) GPT 기반 triple 생성 (part면 스타일 생략)
        triples = build_segment_triples(
            seg_img, seg_id,
            [] if part else global_styles
        )

        # 6) is-a 교체 + part 연결 추가
        triples = [t for t in triples if t["predicate"] != "is-a"]
        triples.insert(0, {
            "subject":  seg_id,
            "predicate":"is-a",
            "object":   part or obj
        })
        if part:
            triples.append({
                "subject": seg_id,
                "predicate": "part-of",
                "object": obj
            })

        # 7) segment 구조로 저장
        out.append({
            "id":      seg_id,
            "path":    path_closed,  # ← 보정된 path
            "object":  obj,
            "part":    part,
            "label":   part or obj,
            "triples": triples
        })

    return out


# ─────────────────────────────────── Segment Labeling ─────────────────────────────────── 

def mask_segment(image: Image.Image, path_str: str) -> Image.Image:
    # 1) SVG path → raster mask
    mask = path_to_mask(path_str, *image.size)          # L-mode (255 = seg)

    # 2) bbox + 패딩
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("RGB", image.size, (255, 255, 255))

    pad = 20                                            # ← 필요 없으면 0
    l, u, r, d = bbox
    l = max(l - pad, 0); u = max(u - pad, 0)
    r = min(r + pad, image.width); d = min(d + pad, image.height)
    bbox_p = (l, u, r, d)

    # 3) 크롭
    segment_crop = image.crop(bbox_p)
    mask_crop    = mask.crop(bbox_p)

    # ── ★ 크기 보정: 간혹 1px 차이 있는 경우 대비 ──
    if mask_crop.size != segment_crop.size:
        mask_crop = mask_crop.resize(segment_crop.size, Image.NEAREST)

    # 4) 흰 배경 컴포지트
    white_bg = Image.new("RGB", segment_crop.size, (255, 255, 255))
    clean_seg = Image.composite(segment_crop, white_bg, mask_crop)
    return clean_seg

# 1) 시스템 메시지 & 함수 스키마  ― enum 없이 자유 라벨
system_msg = {
    "role": "system",
    "content": (
        "You are a vision model that identifies which OBJECT or OBJECT‑PART is "
        "highlighted in a product photo. Answer in lowercase kebab‑case such as "
        "'sofa-armrest', 'table-leg', 'lamp-shade', 'vase', 'background'. "
        "If nothing is selected, return 'background'."
    )
}

FUNCTION_DEF = {
    "name":        "label_segment",
    "description": "Returns a label for the highlighted segment.",
    "parameters": {
        "type":       "object",
        "properties": {
            "label": { "type": "string" }           # ← enum 제거
        },
        "required": ["label"]
    }
}

# -----------------------------------------------------------------
def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf‑8")


def gpt_label_segment(
    pil_crop: Image.Image,
    full_img: Image.Image,
    path_svg: str,
    *,
    model:   str  = "gpt-4o",      # 필요시 "gpt-4o"
    temperature:  float = 0.0,
    top_p:        float = 0.1
) -> str:
    """
    crop + full image + SVG path를 GPT‑4o 에 보내 하이라이트된 부위 라벨을 반환.
    그대로 가져다 쓰는 함수 시그니처·반환값 유지.
    """
    crop_b64 = _img_to_b64(pil_crop)
    full_b64 = _img_to_b64(full_img)

    resp = openai.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        messages=[
            system_msg,
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The SVG path below indicates the highlighted region. "
                            "Return the best matching object or object‑part label."
                        )
                    },
                    {"type": "text",
                     "text": f"[path] {path_svg}…"},   
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{full_b64}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{crop_b64}"}}
                ]
            }
        ],
        tools=[{"type": "function", "function": FUNCTION_DEF}],
        tool_choice={"type": "function",
                     "function": {"name": "label_segment"}}
    )

    args_json = resp.choices[0].message.tool_calls[0].function.arguments
    args = json.loads(args_json) if isinstance(args_json, str) else args_json
    return args["label"]





# ─────────────────────────────────── Knowledge Graph 만들기!!!!!!!─────────────────────────────────── 
VISUALSEM_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a vision+language assistant.  "
        "Given a highlighted product segment, extract *only* its visually‐relevant relations into JSON triples.  "
        "Allowed predicates (relation types) are exactly the 13 VisualSem relations:\n"
        "  • is-a\n"
        "  • has-part\n"
        "  • related-to\n"
        "  • used-for\n"
        "  • used-by\n"
        "  • subject-of\n"
        "  • receives-action\n"
        "  • made-of\n"
        "  • has-property\n"
        "  • gloss-related\n"
        "  • synonym\n"
        "  • part-of\n"
        "  • located-at\n"
        "plus one custom relation **is-style-of** to capture style/adjective information.  "
        "Of all possible relations, output ONLY those describing color, material, texture, shape, function, or style. "
        "You MUST include at least one of each category (color, material, shape, function) if it can be reasonably inferred. "        "\n\n***Always include at least one `is-style-of` triple if any style or adjective applies.***\n\n"
        "Return an object with a single key `triples` mapping to an array of:\n"
        "  {\"subject\": <segment-id>, \"predicate\": <one of the above>, \"object\": <string>}  "
        "Use lowercase kebab-case for predicates and objects."
    )
}

VISUALSEM_FUNCTION_DEF = {
  "name": "return_visual_triples",
  "description": "Return the visual‐attribute triples for this segment.",
  "parameters": {
    "type": "object",
    "properties": {
      "triples": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "subject":   {"type": "string"},
            "predicate": {
              "type": "string",
              "enum": [
                "is-a","has-part","related-to","used-for","used-by",
                "subject-of","receives-action","made-of","has-property",
                "gloss-related","synonym","part-of","located-at","is-style-of"
              ]
            },
            "object":    {"type": "string"}
          },
          "required": ["subject","predicate","object"]
        }
      }
    },
    "required": ["triples"]
  }
}

def image_to_bytes(image, name="image.png"):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = name
    return buffer

def build_segment_triples(
    seg_img: Image.Image,
    seg_id: str,
    global_styles: List[str]
) -> List[Dict]:
    """
    Given a PIL segment crop, call GPT to extract its visualsem‐style triples.
    """
    import io, base64, json, openai

    # 1) encode segment
    buf = io.BytesIO()
    seg_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # 2) ask GPT-4o with our new VisualSem prompt + function
    resp = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            VISUALSEM_SYSTEM_MSG,
            {
                "role": "user",
                "content": [
                    {"type":"text", "text": (
                        f"The subject id is '{seg_id}'.  "
                        "Return *all* applicable visual triples.  "
                        "⚠️ The `is-a` triple MUST use exactly that label."
                        "If multiple values for a predicate exist, include each separately."
                    )},
                    {"type":"image_url", "image_url":{"url":f"data:image/png;base64,{img_b64}"}}
                ]
            }
        ],
        tools=[{"type":"function", "function": VISUALSEM_FUNCTION_DEF}],
        tool_choice={"type":"function","function":{"name":"return_visual_triples"}},
    )

    # 3) pull out the JSON
    call = resp.choices[0].message.tool_calls[0]
    args = call.function.arguments
    parsed = json.loads(args) if isinstance(args, str) else args
    triples = parsed["triples"]

    # ① 전역 스타일 → is-style-of  (이미 있으면 추가 X)
    existing = {(t["predicate"], t["object"]) for t in triples}
    for style in global_styles:
        if ("is-style-of", style) not in existing:
            triples.append({
                "subject": seg_id,
                "predicate": "is-style-of",
                "object": style
            })

    # ── 🔸 완전 중복 제거 (predicate+object 기준) ──
    seen, uniq = set(), []
    for t in triples:
        key = (t["predicate"], t["object"])
        if key not in seen:
            uniq.append(t); seen.add(key)
    return uniq
# ─────────────────────────────────── 세그멘트 기반 생성───────────────────────────────────

# ─────────── NEW: 전체 이미지 스타일 추출 ───────────
# ===== ① 전역 스타일 감지 =====
STYLE_SYS = {
    "role": "system",
    "content": (
        "You are a vision stylist. Summarise the OVERALL visual style "
        "of the given product photo with 1-3 short adjectives "
        "(e.g., 'modern', 'minimalist', 'luxurious'). "
        "Return JSON {\"styles\":[...]} in lowercase kebab-case."
    )
}
STYLE_FDEF = {
  "name": "return_styles",
  "description": "Return overall style adjectives.",
  "parameters": {
    "type": "object",
    "properties": {
      "styles": {
        "type": "array",
        "items": { "type": "string" }
      }
    },
    "required": ["styles"]
  }
}

def _kebab(s: str) -> str:
    """'Soft Vintage' → 'soft-vintage'"""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def detect_image_style(pil_img: Image.Image) -> List[str]:
    """전체 사진에서 1-3개 스타일 형용사 추출"""
    buf = io.BytesIO(); pil_img.save(buf, "PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    resp = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            STYLE_SYS,
            {
              "role":"user",
              "content":[
                {"type":"text","text":"Describe the overall style."},
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64}"}}
              ]
            }
        ],
        tools=[{"type":"function","function":STYLE_FDEF}],
        tool_choice={"type":"function","function":{"name":"return_styles"}}
    )
    args_json = resp.choices[0].message.tool_calls[0].function.arguments
    data = json.loads(args_json) if isinstance(args_json,str) else args_json
    styles = [_kebab(s) for s in data["styles"]]
    return list(dict.fromkeys(styles))[:3]




MIN_SEGMENT_AREA = 0  
IOU_THRESHOLD = 0.1
def path_to_mask(path_d: str, W: int, H: int) -> Image.Image:
    """
    svgpath2mpl 대신 svgpathtools 로 파싱 → 폴리라인으로 샘플링 → Pillow 로 래스터.
    - svgpathtools 은 arc(A)·curve(C/Q) 모두 OK
    """
    # 1) SVG path → Path 객체
    sp_path = parse_path(path_d)

    # 2) Path 길이 기반으로 충분히 조밀하게 포인트 샘플
    seg_len   = sp_path.length()
    n_samples = max(int(seg_len // 1), 600)
    pts = [sp_path.point(t / n_samples) for t in range(n_samples + 1)]
    xy  = [(p.real, p.imag) for p in pts]

    # 3) Pillow (L-mode) 마스크
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(xy, fill=255)

    # 3) Morphological closing → 구멍 메우기
    mask_np = np.array(mask)
    kernel  = np.ones((5, 5), np.uint8)
    closed  = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(closed)


def triples_to_prompt(triples):
    """triples → 콤마 프롬프트 문자열"""
    pieces = []
    for t in triples:
        if t["predicate"].startswith("has-"):
            pieces.append(t["object"])
    # 중복 제거하면서 순서 유지
    return ", ".join(dict.fromkeys(pieces))

# ─────────────────────────────────── remove background ───────────────────────────────────

# ───────── rembg 세션(전역 1회 로딩) ─────────
MODEL_NAME = "isnet-general-use"
REMBG_SESSION = new_session(MODEL_NAME)

# α-matting 파라미터 (필요하면 조정)
MAT_ERODE_SIZE      = 5
MAT_FG_THRESH       = 240
MAT_BG_THRESH       = 10
MAT_BASE_SIZE       = 2048

POST_ERODE_ITER     = 0
POST_GAUSSIAN_SX    = 0
# ───────────────────────────────────────────

def _postprocess_alpha(pil_img: Image.Image) -> Image.Image:
    """알파 채널 침식·블러로 경계 매끈하게"""
    rgba = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    alpha = rgba[..., 3]

    if POST_ERODE_ITER:
        k = np.ones((POST_ERODE_ITER, POST_ERODE_ITER), np.uint8)
        alpha = cv2.erode(alpha, k, iterations=1)

    if POST_GAUSSIAN_SX:
        alpha = cv2.GaussianBlur(alpha, (0, 0),
                                 sigmaX=POST_GAUSSIAN_SX,
                                 sigmaY=POST_GAUSSIAN_SX)

    rgba[..., 3] = alpha
    return Image.fromarray(cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA))

def remove_background(
    pil_img: Image.Image,
    smooth_edges: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    """
    Returns:
        fg_img  : 배경이 투명한 RGBA (Pillow Image)
        fg_mask : 전경=255, 배경=0인 L-mode 마스크
    """
    rgba = remove(
        pil_img,
        session=REMBG_SESSION,
        alpha_matting=True,
        alpha_matting_erode_size=MAT_ERODE_SIZE,
        alpha_matting_foreground_threshold=MAT_FG_THRESH,
        alpha_matting_background_threshold=MAT_BG_THRESH,
        alpha_matting_base_size=MAT_BASE_SIZE,
    )
    if smooth_edges and (POST_ERODE_ITER or POST_GAUSSIAN_SX):
        rgba = _postprocess_alpha(rgba)

    # 전경 / 배경 마스크
    fg_mask = rgba.split()[-1]              # alpha channel (L)
    fg_img  = rgba                          # 그대로 RGBA 반환

    return fg_img, fg_mask

def diff_triples_to_prompt(old_triples, new_triples, target_label=None):
    old_set = set((t["predicate"], t["object"]) for t in old_triples)
    new_set = set((t["predicate"], t["object"]) for t in new_triples)
    diff = new_set - old_set

    def phrase_for(pred, obj):
        if pred == "color":
            return f"change the color to {obj}"
        elif pred == "material":
            return f"use {obj} as the material"
        elif pred == "style":
            return f"apply a {obj} style"
        elif pred == "shape":
            return f"change the shape to {obj}"
        elif pred == "leg-type":
            return f"replace with {obj} legs"
        elif pred == "pattern":
            return f"add a {obj} pattern"
        elif pred == "armrest":
            return f"make the armrest {obj}"
        elif pred == "height":
            return f"adjust the height to be {obj}"
        elif pred == "texture":
            return f"give it a {obj} texture"
        else:
            return f"change {pred} to {obj}"

    phrases = [phrase_for(pred, obj) for pred, obj in diff]
    part_desc = f"the selected part" if not target_label else f"the {target_label}"
    
    if phrases:
        return (
            f"For {part_desc}, " +
            ", then ".join(phrases) + ". " +
            "Please do not modify other parts of the image."
        )
    else:
        return f"Keep the {part_desc} unchanged."


def triples_to_prompt(triples, target_label=None):
    def phrase_for(pred, obj):
        if pred == "color":
            return f"{obj} color"
        elif pred == "material":
            return f"made of {obj}"
        elif pred == "style":
            return f"{obj} style"
        elif pred == "shape":
            return f"{obj} shape"
        elif pred == "leg-type":
            return f"{obj} legs"
        elif pred == "pattern":
            return f"{obj} pattern"
        elif pred == "armrest":
            return f"{obj} armrest"
        elif pred == "height":
            return f"{obj} height"
        elif pred == "texture":
            return f"{obj} texture"
        else:
            return f"{pred}: {obj}"

    phrases = [phrase_for(t["predicate"], t["object"]) for t in triples]
    part_desc = f"the selected part" if not target_label else f"the {target_label}"

    if phrases:
        return (
            f"Design {part_desc} with " +
            ", ".join(phrases) + ". " +
            "Do not modify the rest of the image."
        )
    else:
        return f"Modify only the {part_desc}."


# def infer_triples_from_hierarchical_model_sampling(prompt, graph):
#     def tokenize(text):
#         return re.findall(r'\w+', text.lower())

#     def extract_styles(graph):
#         styles = set()
#         for entry in graph:
#             for seg in entry.get("segments", []):
#                 for tri in seg.get("triples", []):
#                     if tri["predicate"] == "is-style-of":
#                         styles.add(tri["object"].lower())
#         return styles

#     def build_style_priors(graph):
#         style_attr_freq = defaultdict(Counter)
#         for entry in graph:
#             for seg in entry.get("segments", []):
#                 style = next((tri["object"].lower()
#                               for tri in seg["triples"]
#                               if tri["predicate"] == "is-style-of"), None)
#                 if not style:
#                     continue
#                 for tri in seg["triples"]:
#                     if tri["predicate"] not in ("is-style-of", "is-a"):
#                         style_attr_freq[style][(tri["predicate"], tri["object"])] += 1
#         return style_attr_freq

#     tokens = tokenize(prompt)
#     styles = extract_styles(graph)
#     matched_styles = [tok for tok in tokens if tok in styles] or list(styles)
#     priors = build_style_priors(graph)

#     total = Counter()
#     for st in matched_styles:
#         for attr, cnt in priors[st].items():
#             total[attr] += cnt / len(matched_styles)

#     return [{"predicate": pred, "object": obj}
#             for (pred, obj), _ in total.most_common(15)]





def infer_triples_from_hierarchical_model_sampling(
    prompt: str,
    graph: list,
    top_k: int = 15,        # 반환할 triple 수
    alpha: float = 1.0,     # 더 이상 사용되지 않음
    temperature: float = 1.0  # 더 이상 사용되지 않음
):
    """스타일 prior를 이용해 triple 빈도 합산 후 상위 top_k를 반환한다."""

    def tokenize(text: str):
        return re.findall(r'\w+', text.lower())

    def extract_styles(g: list):
        styles = set()
        for entry in g:
            for seg in entry.get("segments", []):
                for tri in seg.get("triples", []):
                    if tri["predicate"] == "is-style-of":
                        styles.add(tri["object"].lower())
        return styles

    def build_style_priors(g: list):
        style_attr_freq = defaultdict(Counter)
        for entry in g:
            for seg in entry.get("segments", []):
                style = next(
                    (tri["object"].lower()
                     for tri in seg.get("triples", [])
                     if tri["predicate"] == "is-style-of"),
                    None
                )
                if not style:
                    continue
                for tri in seg.get("triples", []):
                    if tri["predicate"] not in ("is-style-of", "is-a"):
                        style_attr_freq[style][(tri["predicate"], tri["object"])] += 1
        return style_attr_freq

    tokens = tokenize(prompt)
    styles = extract_styles(graph)
    matched_styles = [tok for tok in tokens if tok in styles] or list(styles)
    priors = build_style_priors(graph)

    total = Counter()
    for st in matched_styles:
        for attr, cnt in priors[st].items():
            total[attr] += cnt / len(matched_styles)

    # 상위 top_k를 predicate/object 형태로 반환
    return [
        {"predicate": pred, "object": obj}
        for (pred, obj), _ in total.most_common(top_k)
    ]

# def infer_triples_from_hierarchical_model_sampling(
#     prompt: str,
#     graph: list,
#     top_k: int = 15,        # 반환할 triple 수
#     alpha: float = 1.0,     # Dirichlet smoothing 강도 (0이면 기존 빈도 그대로)
#     temperature: float = 1.0  # 0<τ<∞, 1=그대로·>1은 평평·<1은 날카로움
# ):
#     """스타일 prior를 이용해 triple 분포를 만들고 확률적으로 샘플링한다."""

#     # ----------------- 유틸 -----------------
#     tokenize = lambda txt: re.findall(r'\w+', txt.lower())

#     def extract_styles(g):
#         styles = set()
#         for entry in g:
#             for seg in entry.get("segments", []):
#                 for tri in seg.get("triples", []):
#                     if tri["predicate"] == "is-style-of":
#                         styles.add(tri["object"].lower())
#         return styles

#     def build_style_priors(g):
#         freq = defaultdict(Counter)
#         for entry in g:
#             for seg in entry.get("segments", []):
#                 style = next((t["object"].lower()
#                               for t in seg["triples"]
#                               if t["predicate"] == "is-style-of"), None)
#                 if not style:
#                     continue
#                 for t in seg["triples"]:
#                     if t["predicate"] not in ("is-style-of", "is-a"):
#                         freq[style][(t["predicate"], t["object"])] += 1
#         return freq
#     # ----------------------------------------

#     tokens = tokenize(prompt)
#     styles = extract_styles(graph)
#     matched = [tok for tok in tokens if tok in styles] or list(styles)

#     priors = build_style_priors(graph)

#     # 1) 스타일별 빈도 → α 스무딩 후 합치기
#     total = Counter()
#     for st in matched:
#         for attr, cnt in priors[st].items():
#             total[attr] += (cnt + alpha) / len(matched)   # 평균 + smoothing

#     if not total:
#         return []      # triple 자체가 없을 수도 있음

#     # 2) 확률 분포 계산 (Soft-max with temperature)
#     attrs, counts = zip(*total.items())
#     probs = np.array(counts, dtype=float)
#     if temperature != 1.0:
#         probs = probs ** (1.0 / temperature)
#     probs /= probs.sum()

#     # 3) 중복 없이 top_k 샘플
#     k = min(top_k, len(attrs))
#     sampled_idx = np.random.choice(len(attrs), size=k, replace=False, p=probs)
#     sampled_attrs = [attrs[i] for i in sampled_idx]

#     return [{"predicate": p, "object": o} for (p, o) in sampled_attrs]


# ─────────────────────────────────── 세션 디렉터리 및 로그 ───────────────────────────────────
ROOT = Path("logs")

def now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def session_dir(user: str, ver: str) -> Path:
    return ROOT / user / ver

# ---------- KG ----------
def _load_json(path: Path, default: Any):
    return json.loads(path.read_text()) if path.exists() else default


def _same_sp(link, subj, pred):
    """subject‑predicate 가 같으면 True (object 는 무시)"""
    return link.get("subject") == subj and link.get("predicate") == pred

def save_full_kg(sess: Path, nodes: list, links: list):
    """
    liked(♥️) 이미지가 변할 때마다 호출.
    전달받은 nodes/links 로 kg_state.json 을 ‘완전히 덮어쓴다’.
    """
    kg_file = sess / "kg_state.json"
    kg_file.parent.mkdir(parents=True, exist_ok=True)
    kg_file.write_text(
        json.dumps({"nodes": nodes, "links": links},
                   ensure_ascii=False, indent=2)
    )



def save_kg_delta(
    sess: Path,
    delta: Union[List[Dict], Dict],          # triple list  or  {nodes,links}
    *,                                       # 명시적 키워드 파라미터
    op: Literal["upsert", "remove"] = "upsert"
):
    """
    클라이언트가 부분적으로 보낸 KG 변화를 세션의 kg_state.json에 반영한다.

    Parameters
    ----------
    delta : list | dict
        • list 형태  →  [{"subject", "predicate", "object"}, …] triple 배열  
        • dict 형태  →  {"nodes":[…], "links":[…]} 부분 그래프
          (links 안도 triple 딕셔너리 형식이어야 함)
    op : {"upsert","remove"}
        • upsert (기본) → 같은 (subject,predicate) 가 있으면 교체, 없으면 추가  
        • remove        → exact match(triple 전부 동일) 를 찾아 삭제
    """
    # kg_file = sess / "kg_state.json"
    cur = _load_json(kg_file, {"nodes": [], "links": []})

    # ─────────────────────────────────────────────────────────────
    # 0)  안전망: 문자열 링크를 dict 로 강제 변환 ("has-color:yellow" → {...})
    # ─────────────────────────────────────────────────────────────
    def _coerce(l):
        if isinstance(l, dict):          # 이미 딕셔너리면 통과
            return l
        if isinstance(l, str):           # "predicate:object" or "predicate=object"
            m = re.match(r"([^:=]+)[:=](.+)", l)
            if m:
                return {"predicate": m.group(1), "object": m.group(2)}
        return None                      # 형식 불명 → 버림

    if isinstance(delta, list):
        delta = [_coerce(x) for x in delta]
        delta = [x for x in delta if x]   # None 제거
    else:  # dict 형태
        delta = {
            "nodes": delta.get("nodes", []),
            "links": [_coerce(x) for x in delta.get("links", []) if _coerce(x)]
        }

    # ---------- 1. links 처리 ------------------------------------
    if isinstance(delta, list):           # triple 리스트
        for t in delta:
            subj, pred, obj = t.get("subject"), t.get("predicate"), t.get("object")
            if not (subj and pred):
                continue                  # 필수 값이 없으면 skip

            if op == "remove":            # 🔹 삭제 모드
                cur["links"] = [l for l in cur["links"]
                                if not (_same_sp(l, subj, pred) and l.get("object") == obj)]
            else:                         # 🔹 upsert 모드
                cur["links"] = [l for l in cur["links"] if not _same_sp(l, subj, pred)]
                cur["links"].append({"subject": subj, "predicate": pred, "object": obj})

    else:  # dict 형태 (nodes / links 둘 다 포함 가능)
        add_unique = lambda lst, item: lst.append(item) if item not in lst else None

        for n in delta["nodes"]:
            add_unique(cur["nodes"], n)

        for l in delta["links"]:
            subj, pred = l.get("subject"), l.get("predicate")
            if not (subj and pred):
                continue

            if op == "remove":
                cur["links"] = [x for x in cur["links"] if not _same_sp(x, subj, pred)]
            else:
                cur["links"] = [x for x in cur["links"] if not _same_sp(x, subj, pred)]
                add_unique(cur["links"], l)

    # ---------- 2. 저장 ------------------------------------------
    # kg_file.parent.mkdir(parents=True, exist_ok=True)
    # kg_file.write_text(json.dumps(cur, ensure_ascii=False, indent=2))
    return cur
# ---------- 이미지 ----------
def _dedup_filename(dst: Path) -> Path:
    """
    dst 가 이미 존재하면 'name-1.ext', 'name-2.ext' … 형태로
    존재하지 않는 첫 번째 파일명을 찾아 반환한다.
    """
    if not dst.exists():
        return dst

    stem, suffix = dst.stem, dst.suffix
    counter = 1
    while True:
        cand = dst.with_name(f"{stem}-{counter}{suffix}")
        if not cand.exists():
            return cand
        counter += 1


def _unique_name(dst: Path) -> Path:
    """
    같은 이름이 있으면 _1, _2, … 숫자를 붙여 충돌을 피한다.
    """
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}_{i}{ext}")
        if not cand.exists():
            return cand
        i += 1


def save_image_file(
    sess: Path,
    fname: str,
    *,
    b64: Optional[str] = None,
    src_path: Optional[str] = None
) -> str:
    """
    세션/images/ 아래에 한‑방향으로만 저장한다.
    이때 fname 안에 슬래시(`sub/dir/file.png`)가 들어와도
    **디렉터리 이름은 무시**하고 basename(`file.png`)만 사용한다.
    """
    images_dir = sess / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # ❶ 디렉터리 부분 제거  ─────────────────────────────
    safe_fname = Path(fname).name.replace("\\", "_")  # 윈도우 백슬러시 보호
    dst = _unique_name(images_dir / safe_fname)

    # ❷ 이미지 데이터가 없으면 (delete / unlike 등) 저장 생략
    if b64 is None and src_path is None:
        return dst.name    # 파일명만 반환

    # ❸ 실제 저장 (Base64 패딩 자동 보정 포함) ────────────
    if b64 is not None:
        # ① Data‑URL 헤더 제거 + 공백·개행 제거
        b64_str = re.sub(r'^data:image/[^;]+;base64,', '', b64)
        b64_str = b64_str.strip().replace('\n', '')

        # ② 잘린 패딩(=) 보정
        b64_str += '=' * (-len(b64_str) % 4)

        # ③ 디코딩 (유효성 검사 포함)
        try:
            img_bytes = base64.b64decode(b64_str, validate=True)
        except binascii.Error as e:
            raise ValueError(f"잘못된 base64 이미지 데이터: {e}") from None

        # ④ 이미지 저장
        img = Image.open(BytesIO(img_bytes)).convert("RGBA")
        img.save(dst)
    else:
        # 로컬 파일 복사
        shutil.copy2(src_path, dst)

    return dst.name

# ---------- 행동 로그 ----------
def append_action(sess: Path, action: dict):
    action.setdefault("ts", now())
    log = sess / "actions.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(action, ensure_ascii=False) + "\n")
        
        import datetime, uuid


import base64, io, re, os
from PIL import Image, ImageDraw
from svgpathtools import parse_path

def mask_by_svgpath(img_b64: str, svg_path: str) -> str:
    """전체 이미지(b64)와 SVG path 문자열을 받아
       해당 영역만 투명 이외 픽셀로 잘라낸 후 b64로 반환."""
    # ① b64 → PIL.Image
    img_data = base64.b64decode(img_b64.split(",")[-1])
    img = Image.open(io.BytesIO(img_data)).convert("RGBA")

    # ② 빈 마스크 생성
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)

    # ③ SVG path → polygon 좌표 추출
    poly = parse_path(svg_path)
    pts = [(seg.start.real, seg.start.imag) for seg in poly]  # 충분히 촘촘하면 OK
    if pts:
        draw.polygon(pts, fill=255)

    # ④ 마스크 적용
    out = Image.new("RGBA", img.size)
    out.paste(img, (0, 0), mask=mask)

    # ⑤ 잘라낸 영역만 최소 bbox로 크롭
    bbox = mask.getbbox()
    if bbox:
        out = out.crop(bbox)

    # ⑥ PIL → b64
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
import csv, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

RowType = Union[Dict[str, Any], List[Any], Tuple[Any, ...], str]

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def _normalize_kg_rows(raw: Any) -> List[Dict[str, Any]]:
    """
    raw를 [{subject, predicate, object, ...}] 리스트 형태로 정규화한다.
    지원 형식:
      - dict: {"subject":..., "predicate":..., "object":..., ...}
      - list/tuple 길이>=3: [sub, pred, obj, ...]
      - str:
          1) JSON 문자열 -> 재귀 파싱
          2) "sub,pred,obj" 또는 탭 구분 -> split 후 dict
    """
    if raw is None:
        return []

    # 1) 리스트 아니면 리스트로 감싸기
    if not isinstance(raw, list):
        raw = [raw]

    out: List[Dict[str, Any]] = []

    def from_list_like(lst: Union[List[Any], Tuple[Any, ...]]) -> Dict[str, Any]:
        d = {
            "subject": _safe_str(lst[0]) if len(lst) > 0 else "",
            "predicate": _safe_str(lst[1]) if len(lst) > 1 else "",
            "object": _safe_str(lst[2]) if len(lst) > 2 else "",
        }
        # 추가 필드가 있다면 extras 로 넣어둔다
        if len(lst) > 3:
            d["extras"] = lst[3:]
        return d

    for r in raw:
        if isinstance(r, dict):
            # 필수키만 보장되게 기본값 채워주기
            out.append({
                "subject":   _safe_str(r.get("subject",   "")),
                "predicate": _safe_str(r.get("predicate", "")),
                "object":    _safe_str(r.get("object",    "")),
                # 선택 필드들 (있으면)
                "role":      _safe_str(r.get("role",      "")),
                "segId":     _safe_str(r.get("segId",     "")),
                "img":       _safe_str(r.get("img",       "")),
                "from":      _safe_str(r.get("from",      "")),
                "to":        _safe_str(r.get("to",        "")),
            })
        elif isinstance(r, (list, tuple)) and len(r) >= 3:
            out.append(from_list_like(r))
        elif isinstance(r, str):
            # 1) JSON 문자열 가능성
            try:
                j = json.loads(r)
                out.extend(_normalize_kg_rows(j))
                continue
            except Exception:
                pass

            # 2) CSV or TSV 형태
            parts = r.split("\t") if "\t" in r else r.split(",")
            if len(parts) >= 3:
                out.append(from_list_like(parts))
            # else: 무시
        # 그 외 타입은 무시
    return out

def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        
        
        
def save_kg_bundle(sess: Path, ts: str, action: str,
                   kg_current: Optional[Dict[str, Any]] = None,
                   kg_delta:   Optional[List[Dict[str, Any]]] = None,
                   seg_b64_map: dict | None = None):
    """
    sess : logs/<user>/<ver> Path
    ts   : ISO time string (":" 대신 "-"로 이미 치환되어 오는 것이 안전)
    action : for file naming clarity (optional)
    """
    safe_ts = ts.replace(":", "-").replace(".", "-")
    base_dir = sess / "kg" / safe_ts
    (base_dir / "segs").mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1) KG current/delta 저장 (image/style 각각)
    if kg_current:
        if "image" in kg_current:
            p = base_dir / "current_image.json"
            p.write_text(json.dumps(kg_current["image"], ensure_ascii=False, indent=2), encoding="utf-8")
            paths["current_image"] = str(p)
        if "style" in kg_current:
            p = base_dir / "current_style.json"
            p.write_text(json.dumps(kg_current["style"], ensure_ascii=False, indent=2), encoding="utf-8")
            paths["current_style"] = str(p)

    if kg_delta:
        if "image" in kg_delta:
            p = base_dir / "delta_image.json"
            p.write_text(json.dumps(kg_delta["image"], ensure_ascii=False, indent=2), encoding="utf-8")
            paths["delta_image"] = str(p)
        if "style" in kg_delta:
            p = base_dir / "delta_style.json"
            p.write_text(json.dumps(kg_delta["style"], ensure_ascii=False, indent=2), encoding="utf-8")
            paths["delta_style"] = str(p)

    # 2) 세그먼트 b64 저장
    if seg_b64_map:
        for seg_id, b64 in seg_b64_map.items():
            seg_id_safe = str(seg_id).replace("/", "_").replace("\\", "_")
            rel = f"kg/{safe_ts}/segs/{seg_id_safe}.png"
            saved = util.save_image_file(sess, rel, b64=b64)
            paths.setdefault("seg_paths", {})[seg_id] = saved

    return paths


def summarize_styles(triples: list, seg_paths: dict):
    """
    triples: [{"predicate":...,"object":...,"segId":...}, ...]
    seg_paths: {seg_id: "kg/.../segs/xxx.png"}  # save_kg_bundle에서 만든 것
    """
    from collections import defaultdict
    # seg → [style], seg → [attr triples]
    seg_styles = defaultdict(list)
    seg_attrs  = defaultdict(list)

    for t in triples or []:
        sid = t.get("segId") or t.get("segment") or t.get("seg_id") or t.get("id")
        if not sid:  # seg 정보 없으면 스킵
            continue
        if t.get("predicate") == "is-style-of":
            seg_styles[sid].append(t.get("object"))
        elif t.get("predicate") not in {"is-a", "label", "name", "has-part"}:
            seg_attrs[sid].append({"predicate": t.get("predicate"), "object": t.get("object")})

    # style → summary
    style_summary = {}
    for sid, styles in seg_styles.items():
        for st in styles:
            entry = style_summary.setdefault(st, {
                "attributes": defaultdict(set),   # pred -> set(objs)
                "segments": []
            })
            # 세그먼트별 속성 리스트
            attrs_for_seg = seg_attrs.get(sid, [])
            for a in attrs_for_seg:
                entry["attributes"][a["predicate"]].add(a["object"])
            entry["segments"].append({
                "id": sid,
                "img_path": seg_paths.get(sid, ""),
                "attributes": attrs_for_seg
            })

    # set → list 로 변환
    for st, ent in style_summary.items():
        ent["attributes"] = {k: sorted(list(v)) for k, v in ent["attributes"].items()}

    return style_summary
