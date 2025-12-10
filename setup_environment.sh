#!/bin/bash
# Setup script for BERTopic analysis environment on Apple Silicon (M4 Pro)
# This script creates and configures the conda environment with GPU support

set -e  # Exit on error

echo "=================================="
echo "BERTopic Environment Setup"
echo "=================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Check available RAM
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    echo "💾 Available RAM: ${TOTAL_RAM}GB"
    if [ "$TOTAL_RAM" -lt 8 ]; then
        echo "⚠️  Warning: Low RAM detected (<8GB)"
        echo "   Recommend using mamba or staged installation"
        echo ""
    fi
fi

# Environment name
ENV_NAME="bertopic"
ENV_FILE="environment_bertopic.yml"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found in current directory"
    exit 1
fi

echo "📋 Environment file: $ENV_FILE"
echo ""

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "   Do you want to remove it and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "ℹ️  Updating existing environment..."
        conda env update -n $ENV_NAME -f $ENV_FILE --prune
        ENV_CREATED=false
    fi
fi

# Create environment if needed
if [ "${ENV_CREATED:-true}" = true ]; then
    echo "🔨 Creating conda environment from $ENV_FILE..."
    echo ""
    
    # Try with mamba first if available
    if command -v mamba &> /dev/null; then
        echo "✅ Mamba found, using for faster solving..."
        mamba env create -f $ENV_FILE
    else
        # Check if libmamba solver is available
        if conda config --show solver 2>/dev/null | grep -q libmamba; then
            echo "✅ Using libmamba solver..."
        else
            echo "💡 Tip: Install mamba for 10-100x faster environment creation:"
            echo "   conda install -n base mamba -c conda-forge"
            echo ""
        fi
        
        # Try to create environment, with fallback on failure
        if ! conda env create -f $ENV_FILE; then
            echo ""
            echo "❌ Environment creation failed (possibly due to memory constraints)"
            echo ""
            echo "🔄 Attempting staged installation as fallback..."
            echo ""
            
            # Create minimal environment
            conda create -n $ENV_NAME python=3.10 -y
            conda activate $ENV_NAME
            
            # Install in stages to reduce memory usage
            echo "📦 Stage 1: Core ML packages..."
            conda install -n $ENV_NAME pytorch torchvision torchaudio -c conda-forge -y
            
            echo "📦 Stage 2: NLP packages..."
            conda install -n $ENV_NAME sentence-transformers transformers -c conda-forge -y
            
            echo "📦 Stage 3: Topic modeling..."
            conda install -n $ENV_NAME bertopic umap-learn hdbscan -c conda-forge -y
            
            echo "📦 Stage 4: Analysis tools..."
            conda install -n $ENV_NAME jupyterlab matplotlib plotly scikit-learn pandas numpy -c conda-forge -y
            
            echo "📦 Stage 5: Additional packages..."
            conda install -n $ENV_NAME spacy nltk gensim langdetect pymongo pymilvus pyarrow tqdm psutil joblib -c conda-forge -y
            
            echo "✅ Staged installation complete"
        fi
    fi
fi

echo ""
echo "✅ Conda environment created/updated successfully"
echo ""

# Activate environment and install additional resources
echo "📦 Installing additional NLP resources..."
echo ""

# Use conda run to execute in the environment
conda run -n $ENV_NAME python -m spacy download en_core_web_sm --quiet

echo "✅ Spacy model downloaded"

conda run -n $ENV_NAME python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

echo "✅ NLTK data downloaded"
echo ""

# Ask about SPECTER2 support
echo "📦 Optional: SPECTER2 model support"
echo "   SPECTER2 models are trained specifically for scientific papers."
echo "   Requires the 'adapters' library (not available via conda)."
read -p "   Install adapters for SPECTER2 support? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔨 Installing adapters library..."
    conda run -n $ENV_NAME pip install adapters>=0.2.1
    echo "✅ Adapters library installed"
else
    echo "ℹ️  Skipping adapters installation (can install later with: pip install adapters)"
fi
echo ""

# Verify GPU availability
echo "🔍 Checking GPU availability..."
echo ""

conda run -n $ENV_NAME python << EOF
import torch
import platform

print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"")

# Check CUDA
if torch.cuda.is_available():
    print("✅ CUDA is available!")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
# Check MPS (Apple Silicon)
elif torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) is available!")
    print("   Your Apple Silicon GPU can be used for acceleration.")
    print(f"   MPS built: {torch.backends.mps.is_built()}")
else:
    print("⚠️  No GPU acceleration available.")
    print("   Models will run on CPU only.")

print(f"")
print(f"CPU cores: {torch.get_num_threads()}")
EOF

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify the installation, run:"
echo "  python -c \"from topic_modeling_analysis import *; print('✅ Module loaded successfully')\""
echo ""
echo "To install SPECTER2 support later (if skipped):"
echo "  conda activate $ENV_NAME"
echo "  pip install adapters>=0.2.1"
echo ""
echo "To run the Jupyter notebook:"
echo "  jupyter lab topic_modeling_notebook.ipynb"
echo ""
echo "For GPU usage tips, see: README_ENVIRONMENT.md"
echo ""
