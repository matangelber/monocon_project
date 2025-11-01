#!/bin/bash

# --- Configuration (STABLE DEPENDENCIES for MMDET3D 0.14.0) ---
ENV_NAME="monocon_A5000_gemini" # Renamed to match your error log
PYTHON_VERSION="3.8"

# PyTorch/CUDA versions compatible with both MMDET3D 0.14.0 and the 'torch.fx' requirement.
TORCH_VERSION="1.8.0"
CUDA_VERSION="11.1"
MMCV_VERSION="1.4.8" # Max version compatible with MMDET3D 0.14.0
MMDET_VERSION="2.11.0"

# --- IMPORTANT: VERIFY THESE LOCAL PATHS ---

MMDET_DIR="/home/matan/Projects/MonoCon/mmdetection-2.11.0"
MMDET3D_DIR="/home/matan/Projects/MonoCon/mmdetection3d-0.14.0"
CUDA_HOME_PATH="/usr/local/cuda-11.1" # Location of your CUDA toolkit installation

echo "--- Starting MMDetection Stable Environment Setup ---"
echo "Targeting PyTorch $TORCH_VERSION, MMCV $MMCV_VERSION, MMDET $MMDET_VERSION, MMDET3D 0.14.0"

# --- 1. Create and Activate Clean Environment ---
conda deactivate
echo "1 - Creating and activating new environment: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source activate $ENV_NAME || { echo "Error activating environment. Exiting."; exit 1; }

# --- 2. Set CUDA Environment Variables (Crucial for Compiling) ---
echo "2 - Setting CUDA environment variables to: $CUDA_HOME_PATH"
export CUDA_HOME=$CUDA_HOME_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# --- 3. Install Core Scientific and Utility Libraries ---
echo "3 - Installing core dependencies (numba, requests, etc.)..."
# *** MODIFICATION 1: Added cython, wheel, and setuptools ***
conda install numba llvmlite numpy pyyaml requests cython wheel setuptools -y

# --- 4. Install PyTorch and Torchvision ---
echo "4 - Installing PyTorch $TORCH_VERSION with CUDA $CUDA_VERSION..."
pip install torch==$TORCH_VERSION torchvision==0.9.0 torchaudio==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu111

# --- 5. Install MMCV-Full (Dependency Bridge) ---
echo "5 - Installing MMCV-Full $MMCV_VERSION, matched to PyTorch $TORCH_VERSION/CUDA $CUDA_VERSION..."
pip install mmcv-full==$MMCV_VERSION -f https://download.openmmlab.com/mmcv/dist/cu$CUDA_VERSION/torch$TORCH_VERSION/index.html

# --- 6. Install MMDetection (Source Install) ---
echo "6 - Installing MMDetection $MMDET_VERSION from source directory: $MMDET_DIR"
if [ -d "$MMDET_DIR" ]; then
    cd "$MMDET_DIR"
    echo "Cleaning up MMDetection build files..."
    rm -rf build dist **/*.egg-info

    echo "Installing MMDetection build-time requirements (cython)..."
    pip install -r requirements/build.txt

    # *** MODIFICATION 2: Explicitly install runtime requirements (mmpycocotools) ***
    #echo "Installing MMDetection runtime requirements (this builds mmpycocotools)..."
    #pip install -r requirements/requirements.txt || { echo "mmpycocotools (from requirements.txt) install failed. Exiting."; exit 1; }

    echo "Installing MMDetection in editable mode..."
    pip install -v -e . || { echo "MMDetection install failed. Exiting."; exit 1; }
    cd -
else
    echo "ERROR: MMDetection directory not found at $MMDET_DIR. Please correct the script path."
    exit 1
fi

# --- 7. Install MMDetection3D (Source Install) ---
echo "7 - Installing MMDetection3D 0.14.0 from source directory: $MMDET3D_DIR"
if [ -d "$MMDET3D_DIR" ]; then
    cd "$MMDET3D_DIR"
    echo "Cleaning up MMDetection3D build files..."
    rm -rf build dist **/*.egg-info
    # Install in editable mode
    pip install -v -e . || { echo "MMDetection3D install failed. Exiting."; exit 1; }
    cd -
else
    echo "ERROR: MMDetection3D directory not found at $MMDET3D_DIR. Please correct the script path."
    exit 1
fi

# --- 8. Install MMsegmentation ---
echo "8 - Installing MMsegmentation 0.13.0"
    pip install mmsegmentation==0.13.0

# --- 9. Final Verification ---
echo "--- Installation Complete! ---"
echo "To run your project, ensure the environment is activated and execute:"
echo "/home/matan/.conda/envs/$ENV_NAME/bin/python /home/matan/Projects/MonoCon/monocon/test_stereo_wrapper.py"