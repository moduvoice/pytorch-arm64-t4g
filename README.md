# pytorch-arm64-t4g

**English** · [한국어](README.ko.md)

**PyTorch CUDA wheels that actually run on the NVIDIA T4G — AWS EC2 `g5g` (Graviton2 arm64 / aarch64 + T4G, compute capability 7.5 / `sm_75`).**

`pip install torch` on a `g5g` instance installs fine, then fails the first time it touches the GPU:

```
UserWarning:
Found GPU0 NVIDIA T4G which is of compute capability (CC) 7.5.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0
...
No published PyTorch CUDA builds for release 2.14.1+cu130 support this GPU.

NVIDIA T4G with CUDA capability sm_75 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_80 sm_90 sm_100 sm_110 sm_120.

torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

This repo is a **pip index** that points at PyTorch builds which do include `sm_75` for aarch64.
Nothing is rebuilt or re-hosted here: the wheels are the ones **AWS publishes** in its public
`framework-binaries` S3 bucket. They are hard to find because the bucket is not a pip index and is barely indexed by
search engines — so this repo makes them installable with one command and documents which ones
really carry `sm_75` kernels.

## Install

Pick a version from the table below and **pin it exactly, including the `+cuXXX` suffix**
(otherwise pip may pick the newer official wheel from PyPI, which has no `sm_75` kernels):

```bash
python3.13 -m venv venv && . venv/bin/activate
pip install "torch==2.12.0+cu132" "torchvision==0.27.0+cu132" \
  --index-url https://moduvoice.github.io/pytorch-arm64-t4g/simple/ \
  --extra-index-url https://pypi.org/simple
```

`--extra-index-url https://pypi.org/simple` is needed for the dependencies (NVIDIA CUDA runtime
packages, numpy, …). Every link in the index carries a `sha256`, so pip verifies the download.

Check it:

```python
import torch
print(torch.cuda.get_arch_list())        # must include 'sm_75'
print((torch.randn(1024, 1024, device="cuda", dtype=torch.float16) @ torch.randn(1024, 1024, device="cuda", dtype=torch.float16)).sum())
```

## Available wheels

Each AWS build exists for one Python version only — use the matching interpreter.

<!-- TABLE -->
| torch | CUDA | Python | torchvision | torchaudio |
|---|---|---|---|---|
| `2.12.0+cu132` | 13.2 | 3.13 | `0.27.0+cu132` | `2.11.0+cu132` |
| `2.11.0+cu130` | 13.0 | 3.13 | `0.26.0+cu130` | `2.11.0+cu130` |
| `2.10.0+cu130` | 13.0 | 3.13 | `0.25.0+cu130` | `2.10.0+cu130` |
| `2.9.0+cu130` | 13.0 | 3.12 | `0.24.0+cu130` | `2.9.0+cu130` |
| `2.8.0+cu129` | 12.9 | 3.12 | `0.23.0+cu129` | `2.8.0+cu129` |
| `2.7.0+cu128` | 12.8 | 3.12 | `0.22.0+cu128` | `2.7.0+cu128` |
| `2.6.0+cu124` | 12.4 | 3.12 | `0.21.0+cu124` | `2.6.0+cu124` |
| `2.6.0+cu126` | 12.6 | 3.12 | `0.21.0+cu126` | `2.6.0+cu126` |
| `2.5.1+cu124` | 12.4 | 3.11 | `0.20.1+cu124` | `2.5.1+cu124` |
| `2.4.0+cu124` | 12.4 | 3.11 | `0.19.0+cu124` | `2.4.0+cu124` |
| `2.3.0+cu121` | 12.1 | 3.11 | `0.18.0+cu121` | `2.3.0+cu121` |
| `2.2.1+cu121` | 12.1 | 3.10 | `0.17.1+cu121` | `2.2.1+cu121` |
<!-- /TABLE -->

## Why the official wheels fail on g5g

PyTorch builds its aarch64 CUDA wheels for data-center Arm systems (Grace Hopper / Grace Blackwell),
so the aarch64 architecture list starts at `sm_80`, while the x86_64 list still includes `sm_75`.
From PyTorch's CI ([`.ci/manywheel/build_env_setup.py`](https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_env_setup.py)):

| CUDA | x86_64 | aarch64 |
|---|---|---|
| 12.6 | 5.0 6.0 7.0 **7.5** 8.0 8.6 9.0 | 8.0 9.0 |
| 13.0 / 13.2 / 13.4 | **7.5** 8.0 8.6 9.0 10.0 12.0 | 8.0 9.0 10.0 11.0 12.0 |

The T4G is Turing (`sm_75`), so it falls through the gap. The AWS builds add `sm_75` to
`libtorch_cuda.so` (verified below).

## How the wheels were checked

Checking a wheel does not need a GPU: [`tools/check_cuda_arch.py`](tools/check_cuda_arch.py) reads the
CUDA fatbinary headers of every shared library in the wheel and lists the `sm_XX` (SASS) and
`compute_XX` (PTX) targets.

```bash
python3 tools/check_cuda_arch.py torch-2.12.0+cu132-cp313-cp313-manylinux_2_28_aarch64.whl
```

Look at **`torch/lib/libtorch_cuda.so`** — that is where PyTorch's own kernels live. Other bundled
libraries (for example `libtorch_nvshmem.so`, or cuBLAS/cuDNN in the CUDA 12 wheels) carry their own
wide architecture lists and can show `sm_75` even when torch itself has none: the official 2.14.1
aarch64 wheel has `sm_75` in `libtorch_nvshmem.so` but not in `libtorch_cuda.so`, and it fails on a T4G.

Results (2026-10-25):

| wheel | `sm_75` in `libtorch_cuda.so` | real T4G (g5g.2xlarge) |
|---|---|---|
| official `torch-2.14.1+cu130` aarch64 | no (`sm_80 sm_90 sm_100 sm_103 sm_110 sm_120 sm_121`) | ❌ `no kernel image is available` |
| official `torch-2.11.0+cu130` aarch64 | no | — |
<!-- VERIFY -->

The per-wheel results (all libraries, sha256) are in [`wheels.json`](wheels.json).
`tools/build_index.py` regenerates the index in `docs/simple/` from it.

## Notes

- **Turing has no native bf16.** Load models in `float16`.
- FlashAttention 2/3 need `sm_80+`; on a T4G use backends that support Turing.
- For vLLM on `g5g`, see the companion recipe [vllm-arm-t4g](https://github.com/moduvoice/vllm-arm-t4g).
- These are AWS's builds, not this repo's. If AWS removes a file, the link breaks — open an issue.

## License

The scripts and docs in this repo are Apache-2.0 ([LICENSE](LICENSE)). PyTorch is BSD-3-Clause;
the wheels are built and hosted by AWS and are not redistributed here.
