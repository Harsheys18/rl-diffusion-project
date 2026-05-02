#!/bin/bash
# Quick start: runs a fast training test to verify everything works
# Should complete in ~10-15 minutes on a 5070 Ti

echo "============================================"
echo "RL-Guided Diffusion - Quick Test Run"
echo "============================================"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

# Check CUDA
"$PYTHON_BIN" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch/CUDA not available. Check installation."
    exit 1
fi

echo ""
echo "Running quick training (50 steps, single prompt)..."
echo ""

"$PYTHON_BIN" train.py \
    --prompt "A green colored rabbit." \
    --max-steps 50 \
    --no-rlg

echo ""
echo "Quick test complete! Check:"
echo "  - logs/ for TensorBoard"  
echo "  - samples/ for generated images"
echo "  - checkpoints/ for model weights"
echo ""
echo "To run full training:"
echo "  python train.py"
echo ""
echo "To evaluate:"
echo "  python evaluate.py --checkpoint ./checkpoints/best"
