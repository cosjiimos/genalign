# Visual Knowledge Graph (VKG) Extraction System

A multimodal AI pipeline for extracting Visual Knowledge Graphs from product images using Semantic-SAM segmentation, GPT-4o vision, and VisualSem ontology.

## Overview

This system automatically:
1. Segments product images into semantic parts
2. Labels each segment with GPT-4o Vision
3. Extracts visual attributes as knowledge graph triples
4. Generates and edits images based on knowledge graph attributes

## Key Features

- **Semantic Segmentation**: Automatic part-level segmentation using Semantic-SAM
- **Visual Attribute Extraction**: GPT-4o-powered extraction of color, material, texture, shape, and style
- **Knowledge Graph Construction**: Structured triples following VisualSem ontology (13 relations + custom style relation)
- **Image Generation & Editing**: GPT-4o image generation with attribute-based control
- **CLIP-based Search**: Semantic image search using CLIP embeddings

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image                               │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │  Background Removal  │ (rembg)
         │   + Preprocessing    │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │   Semantic-SAM       │
         │   Segmentation       │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │  GPT-4o Vision       │
         │  Segment Labeling    │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │  GPT-4o Vision       │
         │  Triple Extraction   │
         │  (VisualSem)         │
         └───────────┬──────────┘
                     │
         ┌───────────▼──────────┐
         │  Visual Knowledge    │
         │      Graph           │
         └──────────────────────┘
```


## VisualSem Ontology Relations

The system uses 13 standard VisualSem relations + 1 custom relation:

| Relation | Description | Example |
|----------|-------------|---------|
| `is-a` | Type/category | seg_0 is-a chair |
| `has-part` | Meronymy | chair has-part armrest |
| `part-of` | Holonymy | armrest part-of chair |
| `made-of` | Material | cushion made-of fabric |
| `has-property` | Attribute | cushion has-property soft |
| `used-for` | Function | chair used-for sitting |
| `used-by` | Agent | desk used-by students |
| `related-to` | Association | sofa related-to living-room |
| `subject-of` | Topic | painting subject-of landscape |
| `receives-action` | Patient | wood receives-action polishing |
| `gloss-related` | Definition | ottoman gloss-related footrest |
| `synonym` | Equivalence | couch synonym sofa |
| `located-at` | Location | lamp located-at desk |
| `is-style-of` | **Custom** | seg_0 is-style-of minimalist |

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenAI API key

### Step 1: Clone Repository
```bash
git clone https://github.com/cosjiimos/GenAlign.git
cd GenAlign
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Model Checkpoints

Download Semantic-SAM checkpoint:
```bash
mkdir -p checkpoints
cd checkpoints
# Download from: https://github.com/UX-Decoder/Semantic-SAM
wget https://huggingface.co/Semantic-SAM/swinl_only_sam_many2many/resolve/main/swinl_only_sam_many2many.pth
cd ..
```

### Step 4: Download CLIP Model (Optional for Offline Use)
```bash
python -c "
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model.save_pretrained('./models/clip-vit-base-patch32')
proc.save_pretrained('./models/clip-vit-base-patch32')
"
```

### Step 5: Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add to `server.py`:
```python
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## Usage

### Start Server
```bash
python server.py
```
Server runs on `http://localhost:5010`

### API Endpoints

#### 1. Initialize Session
```bash
POST /init_session
{
  "user": "username",
  "version": "experiment_v1"
}
```

#### 2. Segment Image & Extract VKG
```bash
POST /segment
{
  "image_b64": "base64_encoded_image",
  "filename": "product.jpg",
  "user": "username",
  "version": "experiment_v1"
}
```

**Response:**
```json
{
  "segments": [
    {
      "id": "seg_0",
      "path": "M 10,20 L 100,200 Z",
      "object": "sofa",
      "part": "armrest",
      "label": "armrest",
      "triples": [
        {"subject": "seg_0", "predicate": "is-a", "object": "armrest"},
        {"subject": "seg_0", "predicate": "made-of", "object": "leather"},
        {"subject": "seg_0", "predicate": "has-property", "object": "brown"}
      ]
    }
  ],
  "result_image_b64": "base64_encoded_segmentation_result"
}
```

#### 3. Generate Image from Attributes
```bash
POST /generate_image_from_attributes
{
  "prompt": "modern table",
  "triples": [
    {"predicate": "made-of", "object": "walnut"},
    {"predicate": "has-property", "object": "rectangular"},
    {"predicate": "is-style-of", "object": "minimalist"}
  ]
}
```

#### 4. Apply Attributes (Image Editing)
```bash
POST /apply_attributes
{
  "base_image_b64": "base64_encoded_image",
  "target_path": "M 10,20 L 100,200 Z",
  "mode": "edit",
  "target_segment": {...},
  "new_triples": [
    {"predicate": "made-of", "object": "chrome"},
    {"predicate": "has-property", "object": "glossy"}
  ]
}
```

#### 5. CLIP-based Search
```bash
POST /search_clip
{
  "query": "modern minimalist chair"
}
```



## Performance

| Operation | Time (GPU) | Time (CPU) |
|-----------|------------|------------|
| Background Removal | ~2s | ~10s |
| Semantic-SAM Segmentation | ~3s | ~30s |
| GPT-4o Labeling (per segment) | ~1s | ~1s |
| GPT-4o Triple Extraction | ~2s | ~2s |
| **Total (5 segments)** | **~12s** | **~55s** |

## Citation

If you use this code in your research, please cite:

```bibtex
@{
}
```

## Acknowledgments

- **Semantic-SAM**: [UX-Decoder/Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM)
- **rembg**: Background removal library
- **OpenAI GPT-4o/GPT-image-1**: Vision and image generation capabilities
- **VisualSem**: Visual semantic relations ontology


## Contact

For questions or issues, please open an issue on GitHub or contact [jiin4900@gmail.com]
