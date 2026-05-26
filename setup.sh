#!/bin/bash
# VKG Extraction System Setup Script

set -e  # Exit on error

echo "=========================================="
echo "VKG Extraction System Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch (CUDA 11.8)
echo -e "\n${YELLOW}Installing PyTorch...${NC}"
read -p "Do you have CUDA GPU? (y/n): " has_cuda
if [ "$has_cuda" = "y" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    echo -e "${GREEN}✓ PyTorch with CUDA support installed${NC}"
else
    pip install torch torchvision
    echo -e "${GREEN}✓ PyTorch CPU version installed${NC}"
fi

# Install other dependencies
echo -e "\n${YELLOW}Installing other dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p checkpoints
mkdir -p models/clip-vit-base-patch32
mkdir -p logs
mkdir -p client/public/img/{sofa_generated,nobg,removed_bg}
mkdir -p debug
echo -e "${GREEN}✓ Directories created${NC}"

# Download Semantic-SAM checkpoint
echo -e "\n${YELLOW}Downloading Semantic-SAM checkpoint...${NC}"
if [ ! -f "checkpoints/swinl_only_sam_many2many.pth" ]; then
    echo "Please download the checkpoint manually from:"
    echo "https://huggingface.co/Semantic-SAM/swinl_only_sam_many2many/resolve/main/swinl_only_sam_many2many.pth"
    echo "Save it to: checkpoints/swinl_only_sam_many2many.pth"
    echo -e "${YELLOW}⚠ Checkpoint download pending${NC}"
else
    echo -e "${GREEN}✓ Checkpoint already exists${NC}"
fi

# Download CLIP model (optional for offline use)
echo -e "\n${YELLOW}Downloading CLIP model...${NC}"
read -p "Download CLIP model for offline use? (y/n): " download_clip
if [ "$download_clip" = "y" ]; then
    python3 << EOF
from transformers import CLIPModel, CLIPProcessor
print("Downloading CLIP model...")
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model.save_pretrained('./models/clip-vit-base-patch32')
proc.save_pretrained('./models/clip-vit-base-patch32')
print("✓ CLIP model downloaded")
EOF
    echo -e "${GREEN}✓ CLIP model downloaded${NC}"
else
    echo -e "${YELLOW}⚠ CLIP model download skipped${NC}"
fi

# Create config file from example
echo -e "\n${YELLOW}Setting up configuration...${NC}"
if [ ! -f "config.py" ]; then
    cp config.example.py config.py
    echo -e "${YELLOW}⚠ Please edit config.py and add your OpenAI API key${NC}"
else
    echo -e "${GREEN}✓ config.py already exists${NC}"
fi

# Setup complete
echo -e "\n=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "=========================================="

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Edit config.py and add your OpenAI API key"
echo "2. Download Semantic-SAM checkpoint if not done:"
echo "   wget -P checkpoints/ https://huggingface.co/Semantic-SAM/swinl_only_sam_many2many/resolve/main/swinl_only_sam_many2many.pth"
echo "3. Activate virtual environment: source venv/bin/activate"
echo "4. Start the server: python server.py"
echo -e "\n${GREEN}Happy VKG extraction!${NC}"
