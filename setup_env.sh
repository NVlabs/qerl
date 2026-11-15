if [ "$(uname -m)" == "aarch64" ]; then
    # aarch64 nodes (e.g. GB200 / GB300): PyPI has no CUDA build of torch and no triton for it, and vLLM only
    # ships aarch64 wheels from 0.10.2 on. flash-attn is not needed (the trainer falls back to SDPA attention).
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
    pytag=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    pip install https://download.pytorch.org/whl/triton-3.4.0-${pytag}-${pytag}-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
    # ptxas that knows the newest GPU architectures (sm_103); picked up automatically by the trainer
    pip install nvidia-cuda-nvcc-cu12==12.9.86 --no-deps
    pip install vllm==0.10.2 transformers==4.56.2 datasets bitsandbytes math-verify==0.5.2 "latex2sympy2_extended>=1.0.6" wandb
else
    GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"
    pip install torch==2.7.1
    pip install torchaudio==2.7.1
    pip install flash-attn==2.7.4.post1 --no-build-isolation
    pip install vllm==0.10.1
fi
pip install trl==0.21.0
pip install peft
# The changes QeRL needs in vLLM and compressed-tensors are applied at import time by
# trl_trainer/runtime_patches.py: nothing is copied into site-packages any more.
pip install compressed-tensors==0.11.0 --no-deps
pip install accelerate==1.10.1 --no-deps
