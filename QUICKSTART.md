# Quick Start Guide

Get up and running with VKG Extraction in 5 minutes!

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- OpenAI API key

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/cosjiimos/GenAlign.git
cd GenAlign

# Run setup script
chmod +x setup.sh
./setup.sh

# Follow the prompts to configure CUDA, download models, etc.
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p checkpoints models/clip-vit-base-patch32 logs

# Download Semantic-SAM checkpoint
wget -P checkpoints/ https://huggingface.co/Semantic-SAM/swinl_only_sam_many2many/resolve/main/swinl_only_sam_many2many.pth

# Copy config example
cp config.example.py config.py
```

## Configuration

Edit `config.py`:

```python
# Add your OpenAI API key
OPENAI_API_KEY = "sk-proj-your-actual-api-key-here"

# (Optional) Change device
DEVICE = "cuda:0"  # or "cpu"
```

Or set environment variable:

```bash
export OPENAI_API_KEY="sk-proj-your-actual-api-key-here"
```

## Start Server

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Start server
python server.py
```

You should see:

```
* Running on http://0.0.0.0:5010
* Debug mode: on
```

## Test Installation

### Method 1: Using Python

Create `test.py`:

```python
import requests
import base64

# Test server
response = requests.get("http://localhost:5010")
print("Server status:", response.status_code)

# Initialize session
init = requests.post("http://localhost:5010/init_session", json={
    "user": "test_user",
    "version": "quickstart"
})
print("Session:", init.json())

# Test segmentation (replace with your image)
with open("test_image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

result = requests.post("http://localhost:5010/segment", json={
    "image_b64": img_b64,
    "filename": "test_image.jpg"
})

print(f"Segments found: {len(result.json()['segments'])}")
for seg in result.json()['segments']:
    print(f"  - {seg['label']}: {len(seg['triples'])} attributes")
```

Run:
```bash
python test.py
```

### Method 2: Using cURL

```bash
# Initialize session
curl -X POST http://localhost:5010/init_session \
  -H "Content-Type: application/json" \
  -d '{"user": "test_user", "version": "quickstart"}'

# Segment image
curl -X POST http://localhost:5010/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
    "filename": "test.jpg"
  }' | jq '.segments[].label'
```

## Example Workflows

### 1. Extract VKG from Product Image

```python
import requests
import base64

# Read image
with open("sofa.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Segment and extract VKG
response = requests.post("http://localhost:5010/segment", json={
    "image_b64": img_b64,
    "filename": "sofa.jpg"
})

segments = response.json()["segments"]

# Print knowledge graph
for seg in segments:
    print(f"\n{seg['label']} ({seg['id']}):")
    for triple in seg['triples']:
        print(f"  {triple['predicate']:20s} → {triple['object']}")
```

### 2. Generate Image from Attributes

```python
import requests
import base64

# Define attributes
triples = [
    {"predicate": "made-of", "object": "walnut"},
    {"predicate": "has-property", "object": "rectangular"},
    {"predicate": "has-property", "object": "dark-brown"},
    {"predicate": "is-style-of", "object": "minimalist"}
]

# Generate image
response = requests.post("http://localhost:5010/generate_image_from_attributes", json={
    "prompt": "modern dining table",
    "triples": triples
})

# Save result
img_data = base64.b64decode(response.json()["full_image_b64"])
with open("generated_table.png", "wb") as f:
    f.write(img_data)

print("Generated image saved to generated_table.png")
```

### 3. Edit Segment Attributes

```python
import requests
import base64

# Load base image
with open("sofa.jpg", "rb") as f:
    base_b64 = base64.b64encode(f.read()).decode()

# First, segment to get paths
seg_response = requests.post("http://localhost:5010/segment", json={
    "image_b64": base_b64,
    "filename": "sofa.jpg"
})

# Get first segment's path
target_path = seg_response.json()["segments"][0]["path"]

# Edit attributes
new_triples = [
    {"predicate": "made-of", "object": "chrome"},
    {"predicate": "has-property", "object": "glossy"}
]

edit_response = requests.post("http://localhost:5010/apply_attributes", json={
    "base_image_b64": base_b64,
    "target_path": target_path,
    "mode": "edit",
    "new_triples": new_triples
})

# Save edited image
img_data = base64.b64decode(edit_response.json()["full_image_b64"])
with open("edited_sofa.png", "wb") as f:
    f.write(img_data)

print("Edited image saved to edited_sofa.png")
```

## Using Example Scripts

The repository includes pre-built example scripts:

```bash
# Run example 1: Segment and extract VKG
python example_usage.py 1

# Run example 2: Generate from attributes
python example_usage.py 2

# Run example 3: Edit segment attributes
python example_usage.py 3

# Run example 4: Full workflow
python example_usage.py 4

# Run example 5: CLIP search
python example_usage.py 5
```

## Viewing Results

All results are saved under `logs/{user}/{version}/`:

```bash
# View segmentation results
ls logs/test_user/quickstart/images/segmented_images/

# View generated images
ls logs/test_user/quickstart/images/generated_images/

# View knowledge graph snapshots
cat logs/test_user/quickstart/kg_snapshots.csv

# View style summaries
cat logs/test_user/quickstart/kg/*/current_style_summary.json
```

## Common Issues

### 1. GPU Out of Memory

**Solution**: Reduce `TEXT_SIZE` in `server.py`:
```python
TEXT_SIZE = 512  # Default: 640
```

### 2. OpenAI API Rate Limit

**Solution**: Add delays between requests or use exponential backoff

### 3. Checkpoint Not Found

**Solution**: Download manually:
```bash
wget -P checkpoints/ https://huggingface.co/Semantic-SAM/swinl_only_sam_many2many/resolve/main/swinl_only_sam_many2many.pth
```

### 4. Port Already in Use

**Solution**: Change port in `server.py`:
```python
app.run(host="0.0.0.0", port=5011, debug=True)
```

## Next Steps

- Read [PIPELINE.md](PIPELINE.md) for technical details
- Check [API.md](API.md) for full API documentation
- See [README.md](README.md) for comprehensive guide

## Getting Help

- GitHub Issues: [https://github.com/cosjiimos/GenAlign/issues](https://github.com/cosjiimos/GenAlign/issues)
- Email: jiin4900@gmail.com

Happy VKG extraction! 🚀
