# Visual Knowledge Graph (VKG) Extraction Pipeline

## Overview

This document provides a detailed technical description of the VKG extraction pipeline.

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT: Product Image                          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Preprocessing      │
                    │   - EXIF Rotation    │
                    │   - RGB Conversion   │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Background Removal  │
                    │  (rembg: ISNet)      │
                    │  Output: RGBA + Mask │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  White Background    │
                    │  Composite (for SAM) │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Semantic-SAM       │
                    │   Segmentation       │
                    │   Output: SVG Paths  │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Postprocess Filter  │
                    │  - IOU Filtering     │
                    │  - Area Filtering    │
                    │  - Path Closing      │
                    └───────────┬──────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
    ┌───────────▼──────────┐        ┌──────────▼─────────┐
    │  GPT-4o Vision       │        │  GPT-4o Vision     │
    │  Segment Labeling    │        │  Style Detection   │
    │  (per segment)       │        │  (whole image)     │
    │  Output: object-part │        │  Output: 1-3 style │
    └───────────┬──────────┘        └──────────┬─────────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  GPT-4o Vision       │
                    │  Triple Extraction   │
                    │  (VisualSem Schema)  │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Triple Assembly     │
                    │  - Add is-a          │
                    │  - Add part-of       │
                    │  - Add global styles │
                    │  - Deduplication     │
                    └───────────┬──────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────┐
│                  OUTPUT: Visual Knowledge Graph                       │
│  {segments: [{id, path, object, part, label, triples}, ...]}        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Steps

### 1. Image Preprocessing

**Function**: `segment()` endpoint → Image loading
**Location**: [server.py:79-106](server.py#L79-L106)

```python
# Decode Base64
img_bytes = base64.b64decode(img_b64)

# Load and auto-rotate based on EXIF
image_orig = ImageOps.exif_transpose(
    Image.open(BytesIO(img_bytes))
).convert("RGB")
```

**Output**:
- RGB PIL Image with correct orientation

---

### 2. Background Removal

**Function**: `util.remove_background()`
**Location**: [server_utils.py:493-518](server_utils.py#L493-L518)

```python
fg_rgba, fg_mask = util.remove_background(
    image_orig,
    smooth_edges=False
)
```

**Process**:
1. Use rembg with ISNet model
2. Apply alpha matting for edge refinement
3. Extract foreground mask (L-mode, 255=foreground, 0=background)

**Output**:
- `fg_rgba`: RGBA image with transparent background
- `fg_mask`: Binary mask (L-mode PIL Image)

**Parameters**:
```python
MODEL_NAME = "isnet-general-use"
MAT_ERODE_SIZE = 5
MAT_FG_THRESH = 240
MAT_BG_THRESH = 10
MAT_BASE_SIZE = 2048
```

---

### 3. White Background Composite

**Function**: Canvas composition
**Location**: [server.py:102-106](server.py#L102-L106)

```python
canvas = Image.new("RGB", fg_rgba.size, (255, 255, 255))
alpha = fg_rgba.split()[-1]
canvas.paste(fg_rgba, mask=alpha)
image_for_seg = canvas
```

**Purpose**:
- Semantic-SAM expects RGB images
- White background improves segmentation quality

---

### 4. Semantic Segmentation (Semantic-SAM)

**Function**: `interactive_infer_image_idino_m2m_auto()`
**Location**: [server.py:110-117](server.py#L110-L117)

```python
with torch.no_grad(), AMP_CTX:
    raw_segments, result_img = interactive_infer_image_idino_m2m_auto(
        model,
        image_for_seg,
        prompt_level,      # [3, 4, 5, 6]
        all_classes,       # []
        all_parts,         # []
        thresh="0.0",
        text_size=TEXT_SIZE,  # 640
        hole_scale=100,
        island_scale=100,
        semantic=True
    )
```

**Output**:
```python
raw_segments = [
    {
        "id": "0",
        "path": "M 120.5,45.2 L 180.3,90.7 ... Z"
    },
    ...
]
```

**Model Details**:
- Architecture: Swin-L Transformer
- Checkpoint: `swinl_only_sam_many2many.pth`
- Config: [configs/semantic_sam_only_sa-1b_swinL.yaml](configs/semantic_sam_only_sa-1b_swinL.yaml)

---

### 5. Postprocessing & Filtering

**Function**: `util.postprocess_segments()`
**Location**: [server_utils.py:51-120](server_utils.py#L51-L120)

#### 5.1 Path Closing
```python
path_closed = ensure_closed_path(seg["path"])
# Adds 'Z' command if missing
```

#### 5.2 SVG Path → Raster Mask
```python
seg_mask_pil = path_to_mask(path_closed, *orig_img.size)
seg_np = (np.array(seg_mask_pil) > 0)
```

**Implementation**: [server_utils.py:426-448](server_utils.py#L426-L448)
- Uses `svgpathtools` to parse SVG paths
- Samples points along curves and arcs
- Renders polygon with PIL ImageDraw
- Applies morphological closing to fill holes

#### 5.3 IOU Filtering
```python
if fg_np is not None:
    inter = np.logical_and(seg_np, fg_np).sum()
    area = seg_np.sum()
    if area < 20 or (inter / area) < IOU_THRESHOLD:
        continue  # Skip background segments
```

**Thresholds**:
```python
IOU_THRESHOLD = 0.1
MIN_SEGMENT_AREA = 20  # pixels
```

---

### 6. Segment Labeling (GPT-4o Vision)

**Function**: `gpt_label_segment()`
**Location**: [server_utils.py:183-231](server_utils.py#L183-L231)

#### Input
1. **Cropped segment** (masked with white background)
2. **Full image** (context)
3. **SVG path** (text description of region)

#### System Prompt
```
You are a vision model that identifies which OBJECT or OBJECT-PART
is highlighted in a product photo. Answer in lowercase kebab-case
such as 'sofa-armrest', 'table-leg', 'lamp-shade', 'vase',
'background'. If nothing is selected, return 'background'.
```

#### Function Schema
```json
{
  "name": "label_segment",
  "parameters": {
    "type": "object",
    "properties": {
      "label": {"type": "string"}
    }
  }
}
```

#### Output
```python
"sofa-armrest"  # object-part format
"table"         # object-only format
```

#### Post-processing
```python
obj, part = (label.split("-", 1) + [None])[:2]
# "sofa-armrest" → obj="sofa", part="armrest"
# "table" → obj="table", part=None
```

---

### 7. Global Style Detection (GPT-4o Vision)

**Function**: `detect_image_style()`
**Location**: [server_utils.py:396-419](server_utils.py#L396-L419)

#### System Prompt
```
You are a vision stylist. Summarise the OVERALL visual style
of the given product photo with 1-3 short adjectives
(e.g., 'modern', 'minimalist', 'luxurious').
Return JSON {"styles":[...]} in lowercase kebab-case.
```

#### Input
- Full product image (after background removal)

#### Output
```python
["modern", "minimalist", "scandinavian"]
```

#### Normalization
```python
def _kebab(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

# "Soft Vintage" → "soft-vintage"
```

---

### 8. Visual Triple Extraction (GPT-4o Vision)

**Function**: `build_segment_triples()`
**Location**: [server_utils.py:303-363](server_utils.py#L303-L363)

#### System Prompt (VISUALSEM_SYSTEM_MSG)
```
You are a vision+language assistant. Given a highlighted product
segment, extract *only* its visually-relevant relations into JSON triples.

Allowed predicates (relation types) are exactly the 13 VisualSem relations:
  • is-a, has-part, part-of, related-to
  • used-for, used-by, subject-of, receives-action
  • made-of, has-property, gloss-related, synonym, located-at

plus one custom relation **is-style-of** to capture style/adjective information.

You MUST include at least one of each category (color, material, shape,
function) if it can be reasonably inferred.

Return an object with a single key `triples` mapping to an array of:
  {"subject": <segment-id>, "predicate": <one of above>, "object": <string>}
```

#### Function Schema
```json
{
  "name": "return_visual_triples",
  "parameters": {
    "type": "object",
    "properties": {
      "triples": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "subject": {"type": "string"},
            "predicate": {
              "type": "string",
              "enum": [
                "is-a", "has-part", "related-to", "used-for",
                "used-by", "subject-of", "receives-action",
                "made-of", "has-property", "gloss-related",
                "synonym", "part-of", "located-at", "is-style-of"
              ]
            },
            "object": {"type": "string"}
          }
        }
      }
    }
  }
}
```

#### Input
1. Segment crop (PNG base64)
2. Segment ID (subject)
3. Global styles (to inject as `is-style-of`)

#### Raw Output Example
```json
{
  "triples": [
    {"subject": "seg_0", "predicate": "is-a", "object": "armrest"},
    {"subject": "seg_0", "predicate": "made-of", "object": "leather"},
    {"subject": "seg_0", "predicate": "has-property", "object": "brown"},
    {"subject": "seg_0", "predicate": "has-property", "object": "curved"},
    {"subject": "seg_0", "predicate": "used-for", "object": "support"}
  ]
}
```

#### Post-processing
```python
# 1. Inject global styles
for style in global_styles:
    if ("is-style-of", style) not in existing:
        triples.append({
            "subject": seg_id,
            "predicate": "is-style-of",
            "object": style
        })

# 2. Deduplicate by (predicate, object)
seen, uniq = set(), []
for t in triples:
    key = (t["predicate"], t["object"])
    if key not in seen:
        uniq.append(t)
        seen.add(key)
```

---

### 9. Triple Assembly & Hierarchy

**Function**: Part of `postprocess_segments()`
**Location**: [server_utils.py:96-108](server_utils.py#L96-L108)

#### Step 1: Remove GPT's `is-a` triple
```python
triples = [t for t in triples if t["predicate"] != "is-a"]
```

#### Step 2: Add correct `is-a` based on label
```python
triples.insert(0, {
    "subject": seg_id,
    "predicate": "is-a",
    "object": part or obj  # part if exists, else object
})
```

#### Step 3: Add `part-of` relation
```python
if part:
    triples.append({
        "subject": seg_id,
        "predicate": "part-of",
        "object": obj
    })
```

#### Example Transformation

**Input** (label = "sofa-armrest"):
```python
obj = "sofa"
part = "armrest"
triples = [
    {"predicate": "is-a", "object": "furniture-component"},  # GPT's guess
    {"predicate": "made-of", "object": "leather"},
    {"predicate": "has-property", "object": "brown"}
]
```

**Output**:
```python
triples = [
    {"subject": "seg_0", "predicate": "is-a", "object": "armrest"},
    {"subject": "seg_0", "predicate": "made-of", "object": "leather"},
    {"subject": "seg_0", "predicate": "has-property", "object": "brown"},
    {"subject": "seg_0", "predicate": "part-of", "object": "sofa"}
]
```

---

### 10. Final Output Structure

**Location**: [server.py:246-251](server.py#L246-L251)

```json
{
  "segments": [
    {
      "id": "seg_0",
      "path": "M 120.5,45.2 L 180.3,90.7 C 200,100 220,95 ... Z",
      "object": "sofa",
      "part": "armrest",
      "label": "armrest",
      "triples": [
        {"subject": "seg_0", "predicate": "is-a", "object": "armrest"},
        {"subject": "seg_0", "predicate": "part-of", "object": "sofa"},
        {"subject": "seg_0", "predicate": "made-of", "object": "leather"},
        {"subject": "seg_0", "predicate": "has-property", "object": "brown"},
        {"subject": "seg_0", "predicate": "has-property", "object": "curved"},
        {"subject": "seg_0", "predicate": "is-style-of", "object": "modern"}
      ]
    },
    {
      "id": "seg_1",
      "path": "M 50,100 L 150,200 ...",
      "object": "sofa",
      "part": "cushion",
      "label": "cushion",
      "triples": [...]
    }
  ],
  "result_image_b64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

---

## Storage & Logging

### CSV Files

#### 1. kg_snapshots.csv
**Purpose**: Time-series log of all triples
**Schema**:
```csv
ts,subject,predicate,object,seg_id,label,seg_path
2025-05-25T10:30:00Z,logs/.../seg_0.png,made-of,leather,seg_0,armrest,logs/.../seg_0.png
```

#### 2. kg_total.csv
**Purpose**: Deduplicated style×attribute combinations
**Schema**:
```csv
ts,style,predicate,object,seg_path,seg_id
2025-05-25T10:30:00Z,modern,made-of,leather,logs/.../seg_0.png,seg_0
```

**Key**: `(style, predicate, object)` – prevents duplicates

---

## Performance Metrics

| Stage | GPU Time | CPU Time | Notes |
|-------|----------|----------|-------|
| Background Removal | ~2s | ~10s | ISNet model |
| Semantic-SAM | ~3s | ~30s | Swin-L backbone |
| GPT-4o Labeling (×5) | ~5s | ~5s | Network I/O bound |
| GPT-4o Triples (×5) | ~10s | ~10s | Network I/O bound |
| **Total (5 segments)** | **~20s** | **~55s** | Parallel GPT calls |

---

## References

- **Semantic-SAM**: [https://github.com/UX-Decoder/Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM)
- **VisualSem**: Visual semantic relations ontology
- **GPT-4o**: OpenAI's multimodal vision model
- **rembg**: [https://github.com/danielgatis/rembg](https://github.com/danielgatis/rembg)
