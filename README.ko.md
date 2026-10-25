# pytorch-arm64-t4g

[English](README.md) · **한국어**

**NVIDIA T4G에서 실제로 도는 PyTorch CUDA 휠 — AWS EC2 `g5g`(Graviton2 arm64 / aarch64 + T4G, compute capability 7.5 / `sm_75`).**

`g5g` 인스턴스에서 `pip install torch`는 설치까지는 되지만, GPU를 처음 쓰는 순간 실패합니다.

```
UserWarning:
Found GPU0 NVIDIA T4G which is of compute capability (CC) 7.5.
...
No published PyTorch CUDA builds for release 2.14.1+cu130 support this GPU.

NVIDIA T4G with CUDA capability sm_75 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_80 sm_90 sm_100 sm_110 sm_120.

torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

이 저장소는 aarch64용이면서 `sm_75`가 들어 있는 PyTorch 빌드를 가리키는 **pip 인덱스**입니다.
여기서 다시 빌드하거나 휠을 다시 올리지 않습니다. 휠은 **AWS가 공개 S3 버킷 `framework-binaries`에 올려 둔 것**입니다.
그 버킷은 pip 인덱스가 아니고 검색에도 거의 잡히지 않아 찾기 어렵기 때문에, 명령 한 줄로 설치할 수 있게 하고
어느 휠에 실제로 `sm_75` 커널이 들어 있는지 검사해 정리했습니다.

## 설치

아래 표에서 버전을 고르고 **`+cuXXX`까지 정확히 고정하세요.** 고정하지 않으면 pip이 PyPI의 더 새로운 공식 휠
(`sm_75` 없음)을 고를 수 있습니다.

```bash
python3.13 -m venv venv && . venv/bin/activate
pip install "torch==2.12.0+cu132" "torchvision==0.27.0+cu132" \
  --index-url https://moduvoice.github.io/pytorch-arm64-t4g/simple/ \
  --extra-index-url https://pypi.org/simple
```

`--extra-index-url https://pypi.org/simple`은 의존성(NVIDIA CUDA 런타임 패키지, numpy 등)을 받는 데 필요합니다.
인덱스의 모든 링크에 `sha256`이 붙어 있어 pip이 내려받은 파일을 검증합니다.

확인:

```python
import torch
print(torch.cuda.get_arch_list())        # 'sm_75'가 있어야 함
print((torch.randn(1024, 1024, device="cuda", dtype=torch.float16) @ torch.randn(1024, 1024, device="cuda", dtype=torch.float16)).sum())
```

## 받을 수 있는 휠

AWS 빌드는 버전마다 파이썬 하나씩만 있습니다. 맞는 파이썬을 쓰세요.

<!-- TABLE -->
| torch | CUDA | 파이썬 | torchvision | torchaudio |
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

## 공식 휠이 g5g에서 안 되는 이유

PyTorch는 aarch64 CUDA 휠을 데이터센터용 Arm 시스템(Grace Hopper / Grace Blackwell)에 맞춰 빌드합니다.
그래서 aarch64 아키텍처 목록은 `sm_80`부터 시작하고, x86_64 목록에는 아직 `sm_75`가 있습니다.
PyTorch CI([`.ci/manywheel/build_env_setup.py`](https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_env_setup.py)) 기준:

| CUDA | x86_64 | aarch64 |
|---|---|---|
| 12.6 | 5.0 6.0 7.0 **7.5** 8.0 8.6 9.0 | 8.0 9.0 |
| 13.0 / 13.2 / 13.4 | **7.5** 8.0 8.6 9.0 10.0 12.0 | 8.0 9.0 10.0 11.0 12.0 |

T4G는 Turing(`sm_75`)이라 이 틈에 빠집니다. AWS 빌드는 `libtorch_cuda.so`에 `sm_75`를 더해 빌드되어 있습니다(아래 검사).

## 검사 방법

휠 검사에는 GPU가 필요 없습니다. [`tools/check_cuda_arch.py`](tools/check_cuda_arch.py)가 휠 안 공유 라이브러리의
CUDA fatbinary 헤더를 읽어 `sm_XX`(SASS)와 `compute_XX`(PTX) 목록을 보여 줍니다.

```bash
python3 tools/check_cuda_arch.py torch-2.12.0+cu132-cp313-cp313-manylinux_2_28_aarch64.whl
```

**`torch/lib/libtorch_cuda.so`** 를 보세요. PyTorch 자체 커널이 여기 있습니다. 함께 들어 있는 다른 라이브러리
(`libtorch_nvshmem.so`, CUDA 12 휠의 cuBLAS·cuDNN 등)는 자기 아키텍처 목록이 넓어서, torch에 `sm_75`가 없어도
`sm_75`가 보일 수 있습니다. 실제로 공식 2.14.1 aarch64 휠은 `libtorch_nvshmem.so`에는 `sm_75`가 있지만
`libtorch_cuda.so`에는 없고, T4G에서 실패합니다.

결과(2026-10-25):

| 휠 | `libtorch_cuda.so`의 `sm_75` | 실제 T4G (g5g.2xlarge) |
|---|---|---|
| 공식 `torch-2.14.1+cu130` aarch64 | 없음 (`sm_80 sm_90 sm_100 sm_103 sm_110 sm_120 sm_121`) | ❌ `no kernel image is available` |
| 공식 `torch-2.11.0+cu130` aarch64 | 없음 | — |
| **이 인덱스** `torch-2.12.0+cu132` (+ `torchvision-0.27.0`, `torchaudio-2.11.0`) | **있음** | ✅ GPU에서 fp16 matmul, conv2d, softmax, torchvision `nms`, torchaudio `lowpass_biquad` |
| **이 인덱스** `torch-2.11.0+cu130` | **있음** | ✅ ([vllm-arm-t4g](https://github.com/moduvoice/vllm-arm-t4g)에서 사용, 2026-06) |
| **이 인덱스**, 표의 나머지 버전 | **있음** | GPU 실행은 안 함 (커널 검사만) |

시험 환경: `g5g.2xlarge`, NVIDIA T4G, 드라이버 595.91, Ubuntu 24.04 arm64, Python 3.13.

휠별 결과(모든 라이브러리, sha256)는 [`wheels.json`](wheels.json)에 있고,
`tools/build_index.py`가 이것으로 `docs/simple/` 인덱스를 다시 만듭니다.

## 참고

- **CUDA 13 휠(`cu130`, `cu132`)은 NVIDIA 드라이버 580 계열 이상이 필요합니다.** 드라이버가 오래됐으면 `cu12x` 휠을 고르세요.
- **Turing에는 bf16 네이티브 지원이 없습니다.** 모델은 `float16`으로 올리세요.
- FlashAttention 2/3는 `sm_80+`가 필요합니다. T4G에서는 Turing을 지원하는 백엔드를 쓰세요.
- `g5g`에서 vLLM은 짝 저장소 [vllm-arm-t4g](https://github.com/moduvoice/vllm-arm-t4g)를 보세요.
- 휠은 이 저장소가 아니라 AWS가 빌드·호스팅합니다. AWS가 파일을 지우면 링크가 깨집니다 — 이슈로 알려 주세요.

## 라이선스

이 저장소의 스크립트와 문서는 Apache-2.0([LICENSE](LICENSE))입니다. PyTorch는 BSD-3-Clause이며,
휠은 AWS가 빌드·호스팅하고 여기서 재배포하지 않습니다.
