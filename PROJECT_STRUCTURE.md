# Project Structure

```
GenAlign/
├── README.md                      # Main documentation
├── QUICKSTART.md                  # Quick start guide
├── PIPELINE.md                    # Technical pipeline details
├── API.md                         # API documentation
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── setup.sh                       # Automated setup script
├── .gitignore                     # Git ignore rules
│
├── config.example.py              # Configuration template
├── server.py                      # Main Flask server
├── server_utils.py                # VKG extraction utilities
├── example_usage.py               # Example scripts
│
├── configs/                       # Model configurations
│   └── semantic_sam_only_sa-1b_swinL.yaml
│
├── semantic_sam/                  # Semantic-SAM model code
│   ├── __init__.py
│   ├── BaseModel.py
│   ├── build_semantic_sam.py
│   │
│   ├── architectures/             # Model architectures
│   │   ├── interactive_mask_dino.py
│   │   ├── build.py
│   │   └── registry.py
│   │
│   ├── backbone/                  # Backbone networks (Swin Transformer)
│   │   ├── swin.py
│   │   ├── swin_new.py
│   │   ├── focal.py
│   │   └── build.py
│   │
│   ├── body/                      # Model body (encoder/decoder)
│   │   ├── general_head.py
│   │   ├── transformer_blocks.py
│   │   │
│   │   ├── encoder/               # Feature encoder
│   │   │   ├── encoder_deform.py
│   │   │   ├── transformer_encoder_fpn.py
│   │   │   └── ops/               # Deformable attention ops
│   │   │
│   │   └── decoder/               # Mask decoder
│   │       ├── interactive_mask_dino.py
│   │       ├── modules.py
│   │       └── utils/
│   │
│   ├── language/                  # Language encoder (if needed)
│   │   ├── encoder.py
│   │   ├── vlpencoder.py
│   │   └── LangEncoder/
│   │       └── transformer.py
│   │
│   ├── modules/                   # Core modules
│   │   ├── criterion_interactive_many_to_many.py
│   │   ├── many2many_matcher.py
│   │   ├── matcher.py
│   │   ├── point_features.py
│   │   ├── position_encoding.py
│   │   └── postprocessing.py
│   │
│   └── utils/                     # Semantic-SAM utilities
│       ├── config.py
│       ├── misc.py
│       └── box_ops.py
│
├── tasks/                         # Inference tasks
│   ├── interactive_idino_m2m_auto.py    # Auto segmentation
│   ├── interactive_idino_m2m.py         # Interactive segmentation
│   ├── interactive_predictor.py         # Predictor interface
│   └── automatic_mask_generator.py      # Mask generation
│
├── utils/                         # General utilities
│   ├── arguments.py               # Argument parsing
│   ├── Config.py                  # Configuration
│   ├── constants.py               # Constants (COCO classes, etc.)
│   ├── distributed.py             # Distributed training utils
│   ├── model.py                   # Model utilities
│   ├── visualizer.py              # Visualization
│   ├── prompt_engineering.py      # Prompt engineering
│   │
│   └── sam_utils/                 # SAM-specific utilities
│       ├── amg.py                 # Automatic mask generation
│       ├── transforms.py          # Image transforms
│       └── onnx.py                # ONNX export
│
├── checkpoints/                   # Model checkpoints (not in repo)
│   └── swinl_only_sam_many2many.pth
│
├── models/                        # Pre-trained models (not in repo)
│   └── clip-vit-base-patch32/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── preprocessor_config.json
│
├── logs/                          # Output logs and results (not in repo)
│   └── {user}/
│       └── {version}/
│           ├── images/
│           │   ├── segmented_images/
│           │   │   └── {timestamp}_{filename}/
│           │   │       ├── seg_result.png
│           │   │       ├── original.png
│           │   │       ├── segments.json
│           │   │       └── segments/
│           │   │           └── seg_{id}_label={label}.png
│           │   │
│           │   └── generated_images/
│           │       └── {timestamp}_{action}/
│           │           ├── full.png
│           │           ├── segments.json
│           │           └── segments/
│           │
│           ├── kg/
│           │   └── {timestamp}_{action}/
│           │       ├── current_graph.json
│           │       ├── current_triples.json
│           │       ├── current_style_summary.json
│           │       ├── delta_graph.json
│           │       ├── delta_triples.json
│           │       ├── snapshots_raw.json
│           │       └── segs/
│           │           └── {seg_id}__{timestamp}.png
│           │
│           ├── kg_snapshots.csv         # All triples with timestamps
│           ├── kg_total.csv             # Deduplicated style×attributes
│           ├── actions_images.csv       # Image action log
│           └── actions.log              # JSON action log
│
├── client/                        # Frontend assets (optional)
│   └── public/
│       └── img/
│           ├── sofa_generated/
│           ├── nobg/
│           └── removed_bg/
│
└── debug/                         # Debug outputs (not in repo)
```

## File Descriptions

### Core Files

| File | Description |
|------|-------------|
| `server.py` | Flask server with 6 main endpoints |
| `server_utils.py` | VKG extraction pipeline functions |
| `config.example.py` | Configuration template |
| `example_usage.py` | Example usage scripts |

### Configuration

| File | Description |
|------|-------------|
| `configs/semantic_sam_only_sa-1b_swinL.yaml` | Semantic-SAM model config |
| `config.py` | User configuration (API keys, paths) |

### Documentation

| File | Description |
|------|-------------|
| `README.md` | Main documentation with installation and usage |
| `QUICKSTART.md` | 5-minute quick start guide |
| `PIPELINE.md` | Detailed technical pipeline explanation |
| `API.md` | Complete API reference |
| `PROJECT_STRUCTURE.md` | This file |

### Key Modules

#### server_utils.py Functions

| Function | Purpose |
|----------|---------|
| `remove_background()` | Remove image background with rembg |
| `postprocess_segments()` | Filter and process SAM segments |
| `gpt_label_segment()` | Label segments with GPT-4o |
| `build_segment_triples()` | Extract attribute triples with GPT-4o |
| `detect_image_style()` | Detect global image style |
| `path_to_mask()` | Convert SVG path to raster mask |
| `ensure_closed_path()` | Close SVG paths |
| `save_kg_delta()` | Save knowledge graph changes |
| `save_image_file()` | Save images to session directory |

#### server.py Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/init_session` | POST | Initialize user session |
| `/segment` | POST | Segment image and extract VKG |
| `/generate_image_from_attributes` | POST | Generate image from attributes |
| `/apply_attributes` | POST | Edit segment attributes |
| `/search_clip` | POST | CLIP-based image search |
| `/log_action` | POST | Log user interactions |

## Data Flow

```
Input Image
    ↓
server.py:/segment
    ↓
server_utils.remove_background()
    ↓
interactive_infer_image_idino_m2m_auto()
    ↓
server_utils.postprocess_segments()
    ├─→ gpt_label_segment()
    ├─→ detect_image_style()
    └─→ build_segment_triples()
    ↓
Output VKG
    ├─→ segments.json
    ├─→ kg_snapshots.csv
    └─→ kg_total.csv
```

## Dependencies

### Core
- PyTorch + torchvision (deep learning)
- transformers (CLIP, language models)
- Flask (web server)
- OpenAI SDK (GPT-4o API)

### Computer Vision
- opencv-python (image processing)
- Pillow (image I/O)
- rembg (background removal)
- scikit-image (advanced processing)

### Semantic-SAM
- fvcore (Facebook Vision core)
- pycocotools (COCO dataset utilities)
- kornia (differentiable CV)
- timm (transformer models)

### Utilities
- svgpathtools (SVG path parsing)
- faiss (vector similarity search)
- pyyaml (config parsing)

## Output Structure

### Segment JSON
```json
{
  "id": "seg_0",
  "path": "M 120.5,45.2 L 180.3,90.7 Z",
  "object": "sofa",
  "part": "armrest",
  "label": "armrest",
  "file_path": "logs/.../seg_0_label=armrest.png",
  "triples": [
    {"subject": "seg_0", "predicate": "is-a", "object": "armrest"},
    {"subject": "seg_0", "predicate": "part-of", "object": "sofa"},
    {"subject": "seg_0", "predicate": "made-of", "object": "leather"}
  ]
}
```

### kg_snapshots.csv
```csv
ts,subject,predicate,object,seg_id,label,seg_path
2025-05-25T10:30:00Z,logs/.../seg_0.png,made-of,leather,seg_0,armrest,logs/.../seg_0.png
```

### kg_total.csv
```csv
ts,style,predicate,object,seg_path,seg_id
2025-05-25T10:30:00Z,modern,made-of,leather,logs/.../seg_0.png,seg_0
```

## Git Structure

### Tracked Files
- Source code (`.py`)
- Documentation (`.md`)
- Configuration examples (`.example.*`)
- Dependencies (`requirements.txt`)
- Setup scripts (`.sh`)

### Ignored Files (see .gitignore)
- Logs and data (`logs/`)
- Model checkpoints (`checkpoints/`, `*.pth`)
- Generated images (`client/public/img/`)
- Python cache (`__pycache__/`, `*.pyc`)
- Virtual environment (`venv/`)
- API keys (`.env`, `config.py`)
- CLIP index files (`*.index`, `*.pkl`)
