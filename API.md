# API Documentation

## Base URL
```
http://localhost:5010
```

## Endpoints

### 1. Initialize Session

Initialize a new user session for logging and data organization.

**Endpoint**: `POST /init_session`

**Request Body**:
```json
{
  "user": "researcher_name",
  "version": "experiment_v1"
}
```

**Response**:
```json
{
  "status": "ok",
  "path": "logs/researcher_name/experiment_v1"
}
```

**Notes**:
- `version` is optional. If not provided, uses current timestamp
- Creates directory structure under `logs/{user}/{version}/`

---

### 2. Segment Image & Extract VKG

Segment a product image and extract visual knowledge graph triples.

**Endpoint**: `POST /segment`

**Request Body**:
```json
{
  "image_b64": "data:image/png;base64,iVBORw0KGgo...",
  "filename": "product.jpg",
  "user": "researcher_name",
  "version": "experiment_v1"
}
```

**Parameters**:
- `image_b64` (required): Base64-encoded image (PNG/JPG)
- `filename` (optional): Original filename for logging
- `user` (optional): Override session user
- `version` (optional): Override session version

**Response**:
```json
{
  "segments": [
    {
      "id": "seg_0",
      "path": "M 120.5,45.2 L 180.3,90.7 Z",
      "object": "sofa",
      "part": "armrest",
      "label": "armrest",
      "file_path": "logs/.../seg_0_label=armrest.png",
      "triples": [
        {
          "subject": "seg_0",
          "predicate": "is-a",
          "object": "armrest"
        },
        {
          "subject": "seg_0",
          "predicate": "part-of",
          "object": "sofa"
        },
        {
          "subject": "seg_0",
          "predicate": "made-of",
          "object": "leather"
        },
        {
          "subject": "seg_0",
          "predicate": "has-property",
          "object": "brown"
        },
        {
          "subject": "seg_0",
          "predicate": "is-style-of",
          "object": "modern"
        }
      ]
    }
  ],
  "result_image_b64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Saved Files**:
```
logs/{user}/{version}/images/segmented_images/{timestamp}_{filename}/
├── seg_result.png          # Segmentation visualization
├── original.png            # Original image
├── segments.json           # Full segment data
└── segments/
    ├── seg_0_label=armrest_{timestamp}.png
    └── seg_1_label=cushion_{timestamp}.png
```

**CSV Logs**:
- `kg_snapshots.csv`: All triples with timestamps
- `kg_total.csv`: Deduplicated style×attribute combinations

---

### 3. Generate Image from Attributes

Generate a new product image based on text prompt and attribute triples.

**Endpoint**: `POST /generate_image_from_attributes`

**Request Body**:
```json
{
  "prompt": "modern dining table",
  "triples": [
    {"predicate": "made-of", "object": "walnut"},
    {"predicate": "has-property", "object": "rectangular"},
    {"predicate": "has-property", "object": "dark-brown"},
    {"predicate": "is-style-of", "object": "minimalist"}
  ],
  "graph": [],
  "top_k": 15,
  "alpha": 1.0,
  "temperature": 1.0,
  "user": "researcher_name",
  "version": "experiment_v1"
}
```

**Parameters**:
- `prompt` (required): Text description of object
- `triples` (optional): Explicit attribute triples
- `graph` (optional): Full KG for Bayesian inference (if triples empty)
- `top_k` (optional): Number of attributes to infer (default: 15)
- `alpha` (optional): Smoothing parameter (default: 1.0)
- `temperature` (optional): Sampling temperature (default: 1.0)

**Response**:
```json
{
  "full_image_path": "http://localhost:5010/generated/gen_abc123.png",
  "full_image_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "full_segments": [
    {
      "id": "seg_0",
      "path": "M 50,100 L 150,200 Z",
      "object": "table",
      "part": "tabletop",
      "label": "tabletop",
      "file_path": "logs/.../seg_0_label=tabletop.png",
      "triples": [...]
    }
  ]
}
```

**Saved Files**:
```
logs/{user}/{version}/images/generated_images/{timestamp}_from_prompt/
├── full.png
├── segments.json
└── segments/
    └── seg_0_label=tabletop.png
```

---

### 4. Apply Attributes (Image Editing)

Edit a specific segment in an existing image by changing its attributes.

**Endpoint**: `POST /apply_attributes`

**Request Body**:

#### Mode 1: Edit (Modify Existing Segment)
```json
{
  "base_image_b64": "data:image/png;base64,...",
  "target_path": "M 120.5,45.2 L 180.3,90.7 Z",
  "mode": "edit",
  "target_segment": {
    "id": "seg_0",
    "triples": [
      {"predicate": "made-of", "object": "leather"},
      {"predicate": "has-property", "object": "brown"}
    ]
  },
  "new_triples": [
    {"predicate": "made-of", "object": "chrome"},
    {"predicate": "has-property", "object": "glossy"}
  ]
}
```

#### Mode 2: Transfer (Copy Style from Another Segment)
```json
{
  "base_image_b64": "data:image/png;base64,...",
  "target_path": "M 120.5,45.2 L 180.3,90.7 Z",
  "mode": "transfer",
  "source_segment": {
    "id": "seg_1",
    "bbox": {
      "x": 100,
      "y": 50,
      "width": 200,
      "height": 150
    },
    "triples": [
      {"predicate": "made-of", "object": "velvet"},
      {"predicate": "has-property", "object": "navy-blue"}
    ]
  }
}
```

**Parameters**:
- `base_image_b64` (required): Original image
- `target_path` (required): SVG path of segment to edit
- `mode` (required): "edit" or "transfer"
- `target_segment` (edit mode): Segment to modify
- `source_segment` (transfer mode): Reference segment
- `new_triples` (edit mode): New attribute triples

**Response**:
```json
{
  "full_image_path": "http://localhost:5010/generated/full_xyz789.png",
  "full_image_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "full_segments": [...]
}
```

**Saved Files**:
```
logs/{user}/{version}/images/generated_images/{timestamp}_apply/
├── full.png
├── original.png
├── segments.json
└── segments/
```

---

### 5. CLIP-based Image Search

Search for similar images using CLIP text-image similarity.

**Endpoint**: `POST /search_clip`

**Request Body**:
```json
{
  "query": "modern minimalist chair"
}
```

**Response**:
```json
[
  "logs/user1/exp1/images/seg_0.png",
  "logs/user2/exp3/images/seg_5.png",
  "logs/user1/exp2/images/seg_12.png",
  ...
]
```

**Notes**:
- Returns paths ranked by CLIP similarity
- Requires pre-built FAISS index (`clip_image_index.index`)
- Build index with: `python create_clip_index.py`

---

### 6. Log Action

Log user interactions and knowledge graph changes.

**Endpoint**: `POST /log_action`

**Request Body**:
```json
{
  "user": "researcher_name",
  "version": "experiment_v1",
  "action": "edit_segment",
  "before_image": {
    "b64": "data:image/png;base64,..."
  },
  "after_image": {
    "b64": "data:image/png;base64,..."
  },
  "segment": {
    "id": "seg_0",
    "path": "M 120.5,45.2 L 180.3,90.7 Z",
    "b64": "data:image/png;base64,..."
  },
  "prompt": "Change material to chrome",
  "triple": [
    {
      "predicate": "made-of",
      "object": "chrome",
      "role": "new",
      "segId": "seg_0",
      "b64": "data:image/png;base64,..."
    }
  ],
  "kg_current": {
    "graph": [...],
    "triples": [...],
    "image": {...},
    "style": {...}
  },
  "kg_delta": {
    "graph": [...],
    "triples": [...]
  },
  "segments_b64": {
    "seg_0": "data:image/png;base64,...",
    "seg_1": "data:image/png;base64,..."
  },
  "snapshots": {
    "image_full": {...},
    "style": {...}
  }
}
```

**Response**:
```json
{
  "ok": true,
  "kg_files": {
    "current_graph": "logs/.../kg/.../current_graph.json",
    "current_triples": "logs/.../kg/.../current_triples.json",
    "current_style_summary": "logs/.../kg/.../current_style_summary.json",
    "seg_paths": {
      "seg_0": "kg/.../segs/seg_0__2025-05-25T10-30-00Z.png"
    }
  }
}
```

**Saved Files**:
```
logs/{user}/{version}/
├── kg/
│   └── {timestamp}_{action}/
│       ├── current_graph.json
│       ├── current_triples.json
│       ├── current_style_summary.json
│       ├── delta_graph.json
│       ├── delta_triples.json
│       └── segs/
│           ├── seg_0__{timestamp}.png
│           └── seg_1__{timestamp}.png
├── images/
│   ├── {timestamp}_{action}_before.png
│   └── {timestamp}_{action}_after.png
├── actions.log            # JSON log of all actions
└── actions_images.csv     # CSV summary
```

---

## Error Responses

All endpoints return error responses in the following format:

```json
{
  "error": "Error message description"
}
```

**HTTP Status Codes**:
- `400`: Bad Request (missing parameters)
- `500`: Internal Server Error

---

## Data Types

### Segment Object
```typescript
{
  id: string                 // "seg_0", "seg_1", ...
  path: string               // SVG path string
  object: string             // "sofa", "table", ...
  part: string | null        // "armrest", "leg", null
  label: string              // "armrest" or "table"
  file_path: string          // File path to segment image
  triples: Triple[]          // Visual attribute triples
}
```

### Triple Object
```typescript
{
  subject: string            // Segment ID
  predicate: string          // Relation type
  object: string             // Attribute value
}
```

**Allowed Predicates**:
- `is-a`: Type/category
- `has-part`: Meronymy
- `part-of`: Holonymy
- `made-of`: Material
- `has-property`: Attribute
- `used-for`: Function
- `used-by`: Agent
- `related-to`: Association
- `subject-of`: Topic
- `receives-action`: Patient
- `gloss-related`: Definition
- `synonym`: Equivalence
- `located-at`: Location
- `is-style-of`: Style/adjective

---

## Rate Limits

OpenAI API rate limits apply:
- GPT-4o Vision: ~10 requests/minute (varies by tier)
- Image Generation: ~5 images/minute

**Recommendation**: Implement exponential backoff for production use.

---

## Example cURL Requests

### Segment Image
```bash
curl -X POST http://localhost:5010/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "data:image/png;base64,iVBORw0KGgo...",
    "filename": "sofa.jpg"
  }'
```

### Generate Image
```bash
curl -X POST http://localhost:5010/generate_image_from_attributes \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "modern table",
    "triples": [
      {"predicate": "made-of", "object": "walnut"},
      {"predicate": "has-property", "object": "rectangular"}
    ]
  }'
```

### Search Images
```bash
curl -X POST http://localhost:5010/search_clip \
  -H "Content-Type: application/json" \
  -d '{"query": "modern minimalist chair"}'
```
