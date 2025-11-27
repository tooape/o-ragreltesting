#!/bin/bash
# Lambda Cloud Setup Script for O-RAG Benchmark
#
# Run this on a fresh Lambda instance to set up the environment.
#
# Usage:
#   chmod +x lambda_setup.sh
#   ./lambda_setup.sh

set -e

echo "=========================================="
echo "O-RAG Benchmark - Lambda Setup"
echo "=========================================="

# Check if running on Lambda (CUDA available)
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Are you on a Lambda instance?"
fi

# System info
echo ""
echo "System Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "No GPU detected"
echo ""

# Create workspace
WORKSPACE="/home/ubuntu/o-rag"
PERSISTENT_FS="/home/ubuntu/persistent"

mkdir -p $WORKSPACE
mkdir -p $PERSISTENT_FS/embeddings
mkdir -p $PERSISTENT_FS/results
mkdir -p $PERSISTENT_FS/logs

cd $WORKSPACE

# Clone repo if not exists
if [ ! -d "o-ragreltesting" ]; then
    echo "Cloning repository..."
    git clone https://github.com/tooape/o-ragreltesting.git
else
    echo "Repository exists, pulling latest..."
    cd o-ragreltesting
    git pull
    cd ..
fi

cd o-ragreltesting

# Create virtual environment
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers: OK')"
python -c "from rank_bm25 import BM25Okapi; print('rank-bm25: OK')"

# Create symlinks to persistent storage
ln -sf $PERSISTENT_FS/embeddings data/embeddings 2>/dev/null || true
ln -sf $PERSISTENT_FS/results results 2>/dev/null || true

# Download model (warm cache)
echo ""
echo "Pre-downloading EmbeddingGemma model..."
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading google/embedding-gemma-001...')
model = SentenceTransformer('google/embedding-gemma-001')
print('Model downloaded successfully!')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Workspace: $WORKSPACE/o-ragreltesting"
echo "Persistent storage: $PERSISTENT_FS"
echo ""
echo "To run benchmark:"
echo "  cd $WORKSPACE/o-ragreltesting"
echo "  source venv/bin/activate"
echo "  python scripts/run_benchmark.py --chunks data/vault_chunks.json --qrels data/qrels.json"
echo ""
echo "To run in background with logging:"
echo "  nohup python scripts/run_benchmark.py > $PERSISTENT_FS/logs/benchmark.log 2>&1 &"
echo ""
