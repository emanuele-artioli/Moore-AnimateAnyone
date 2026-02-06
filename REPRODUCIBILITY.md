# Reproducibility & Fixes Report

This document records the fixes we applied to make `Moore-AnimateAnyone` run cleanly on the current machine and how to reproduce them. It focuses on items that do not show up as normal source code changes (environment fixes, site-package edits, and manual installs) and lists commands and locations so they can be reproduced or rolled into environment setup scripts.

---

## Summary of changes

1. Code-level (versioned in repo)
   - Added targeted warning suppression in `src/dwpose/__init__.py` to silence noisy timm deprecation warnings during runtime.

2. Site-packages edits (NOT committed to git; applied directly to env's site-packages)
   - Replaced deprecated imports from `timm.models.layers` / `timm.models.registry` to the newer locations in the following packages (in the active conda env `animate-anyone`):
     - `controlnet_aux/segment_anything/modeling/tiny_vit_sam.py` — switched to `from timm.layers import ...` and `from timm.models import register_model`.
     - `open_clip/timm_model.py` — switched to `from timm.layers import ...`.
     - `xformers/benchmarks/benchmark_transformer.py` — switched to `from timm.layers import ...`.
     - `controlnet_aux/zoe/.../dpt_depth.py` — switched to `from timm.layers import get_act_layer`.
   - Rationale: these changes remove FutureWarnings that are emitted when importing packages that still reference the old timm internal paths.
   - Recommendation: open PRs in those upstream repos to fix imports — site-package edits are a local workaround.

3. Package upgrades in the environment
   - Upgraded packages in the env `animate-anyone`:
     - `controlnet_aux` -> 0.0.10
     - `segment-anything` -> 1.0
   - Note: these upgrades were performed with `pip install -U` inside the conda env.

4. Please-upgrade (requirements changes)
   - Updated `requirements.txt` in the repo:
     - `controlnet-aux==0.0.10`
     - `segment-anything==1.0`
     - Added NVIDIA runtime packages (to enable CUDA 12 / cuDNN 9 support using manylinux wheels):
       - `nvidia-cuda-runtime-cu12`
       - `nvidia-cudnn-cu12`
       - `nvidia-cublas-cu12`
       - `nvidia-curand-cu12`
       - `nvidia-cufft-cu12`
   - Rationale: these wheels install the shared libs (`libcublasLt.so.12`, `libcurand.so.*`, `libcufft.so.*`, etc.) into the environment without needing sudo.

5. Environment activation scripts (NOT in git; created in conda env)
   - To make the NVIDIA runtime libraries available (so ONNXRuntime CUDA provider can find them), created activation scripts inside the conda env:
     - `$(CONDA_PREFIX)/etc/conda/activate.d/animate_anyone_env_vars.sh` — prepends relevant `nvidia` lib directories to `LD_LIBRARY_PATH` when the env is activated.
     - `$(CONDA_PREFIX)/etc/conda/deactivate.d/animate_anyone_env_vars.sh` — restores original `LD_LIBRARY_PATH` when env is deactivated.
   - Path on this machine (example):
     - `/home/itec/emanuele/.conda/envs/animate-anyone/etc/conda/activate.d/animate_anyone_env_vars.sh`

6. ONNXRuntime / CUDA fixes
   - Observed error: ONNXRuntime failed to load `libonnxruntime_providers_cuda.so` due to missing shared objects (examples: `libcublasLt.so.12`, `libcurand.so.10`, `libcufft.so.11`).
   - Fixed by installing the NVIDIA manylinux wheels into the environment:
     - `pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cufft-cu12`
   - After adding those libs and setting `LD_LIBRARY_PATH` via the activation script, ONNXRuntime successfully created a CUDA provider session.
   - Note: ONNXRuntime wheel must match CUDA version; ensure ONNXRuntime GPU is compatible with CUDA 12 / cuDNN 9 if you need a GPU-optimized ONNX wheel.

---

## Reproduction steps (fresh machine)

1. Create and activate the environment (example with conda):

```bash
conda create -n animate-anyone python=3.10 -y
conda activate animate-anyone
```

2. Install normal python requirements (this repo) - ensure `requirements.txt` is up-to-date:

```bash
pip install -r requirements.txt
# If you need the exact versions used in our env, pin the installed versions in your own lockfile
```

3. Install NVIDIA runtime libraries (these are in `requirements.txt` now; if not, run):

```bash
pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cufft-cu12
```

4. Add the environment activation scripts (example shown in this repo's notes — the assistant created them automatically in the conda env used here). If you need them in your env, create the files below and adjust paths to match your `$CONDA_PREFIX`:

```
$CONDA_PREFIX/etc/conda/activate.d/animate_anyone_env_vars.sh
# exports LD_LIBRARY_PATH to include:
# $CONDA_PREFIX/lib
# $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib
# $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib
# $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib
# $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/curand/lib
# $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cufft/lib
```

5. (Optional) If you still see the timm deprecation warnings when importing some third-party packages, either:
   - Update those third-party packages to versions that import `timm.layers` properly, or
   - Open PRs for those projects to change `from timm.models.layers` -> `from timm.layers`, and `from timm.models.registry` -> `from timm.models`.

6. (Optional, local workaround) If immediate silence of registration warnings is desired, add a targeted `warnings.filterwarnings(...)` in your entry code (we added a targeted one in `src/dwpose/__init__.py`). This is a pragmatic but temporary fix.

---

## Notes & recommended follow-ups

- Upstream PRs: We edited site-packages locally to remove deprecated import paths. Please open PRs to upstream repos (`controlnet_aux`, `open_clip`, `xformers`, etc.) with the small import changes so future installs don't require local patches.
- Lockfile: Consider adding environment YAML (conda env export or pinned `pip freeze`) to make reproduction perfectly deterministic — the repo `requirements.txt` is now updated but a lockfile ensures exact versions.
- ONNXRuntime: Ensure you use an `onnxruntime` / `onnxruntime-gpu` build that matches your CUDA+cuDNN versions when deploying to other machines (the GPU provider is sensitive to ABI mismatch).

---

If you want, I can:
- Draft PRs for the upstream fixes and prepare diffs to submit, and
- Add an example `environment.yml` or `conda-lock` file to the repo to capture exact package versions used here.

