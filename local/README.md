# 이 컴퓨터 (`hanbin5/local`)

이 브랜치는 `main`에 이 컴퓨터에서만 의미 있는 설정을 더한 것입니다. 재현 기록과 코드는 `main`에 있습니다([`repro/README.md`](../repro/README.md)). 이 브랜치는 `local/`만 더하므로 `main`을 merge해도 충돌하지 않습니다.

```bash
git merge main
```

## 장비

| 항목 | 값 |
|---|---|
| 호스트 / 사용자 | `super` / `inner` |
| CPU, RAM | i9-12900K, 31 GiB |
| GPU | RTX 3060 12 GB (드라이버 580.178.04) |
| OS | Ubuntu 24.04.5 LTS (커널 6.8) |
| 디스크 | `/` 435 GB (nvme0n1p8), `/data` 245 GB (nvme0n1p9, 2026-10-16 추가) |

## 체크아웃 두 개

- `/home/inner/real2sim/SimFoundry`: **실행용**입니다.
  - 여기에 `deps/`(외부 저장소들), `Data`, `checkpoints`, 응답 캐시 `.cache/simfoundry/`, `api_keys.txt`가 있습니다.
  - conda 환경 `simfoundry`에는 이 폴더의 `simfoundry`와 `deps/BEHAVIOR-1K/OmniGibson`이 editable로 설치돼 있습니다. 그래서 폴더를 옮기면 환경이 깨집니다.
  - 원격 `origin`이 SSH 주소인데 이 머신에는 SSH 키가 없습니다.
- `/home/inner/real2sim/innerinner`: git 정리용 clone(HTTPS)입니다.
  - `deps/`가 없어서 파이프라인은 못 돌립니다.
  - `local/link_data.sh`로 `Data`와 `checkpoints` 링크를 만들면 CPU 도구(`repro/compare_scenes.py`, `repro/eval_pose_fit.py`, `repro/compose_real_vs_sim.py`)는 돌릴 수 있습니다.

## 큰 파일은 `/data`에 (2026-10-16)

루트 디스크를 비우려고 데이터와 가중치를 `/data`로 옮기고, 원래 자리에 심볼릭 링크를 두었습니다. 전체 목록은 [`MOVED_FROM.txt`](MOVED_FROM.txt)에 있습니다(다른 프로젝트 `acwm` 포함). 이 저장소 쪽은 다음과 같습니다.

| 원래 자리 | `/data` |
|---|---|
| `<checkout>/Data` | `/data/datasets/simfoundry` |
| `<checkout>/checkpoints` | `/data/models/simfoundry-checkpoints` (SAM 2.1) |
| `<checkout>/deps/void-model/{CogVideoX-Fun-V1.5-5b-InP,void_pass1,void_pass2}` | `/data/models/void-model/` |
| `<checkout>/deps/ml-depth-pro/checkpoints` | `/data/models/ml-depth-pro-checkpoints` |
| `<checkout>/deps/BEHAVIOR-1K/datasets/{omnigibson-robot-assets,real2sim-assets,custom-assets}` | `/data/datasets/behavior-1k/` |
| `~/.cache/{huggingface/hub,hy3dgen,torch}` | `/data/models/{huggingface/hub,hy3dgen,torch}` |
| `/home/inner/real2sim/tools/{ollama_models,checkpoints}` | `/data/models/{ollama,real2sim-tools-checkpoints}` (Ollama 모델, SAM 3.1 가중치) |

새로 clone하거나 `deps/`를 다시 설치했다면 링크를 다시 만듭니다. 맞는 링크는 그대로 두고, 없는 링크만 만들고, 진짜 파일이나 폴더가 있으면 건드리지 않고 알립니다.

```bash
local/link_data.sh --dry-run /home/inner/real2sim/SimFoundry
```

```bash
local/link_data.sh /home/inner/real2sim/SimFoundry
```

## 환경과 서비스

`repro/`의 래퍼는 이 체크아웃의 [`env.sh`](env.sh)를 먼저 읽습니다. 그래서 아래 경로들을 따로 export하지 않아도 됩니다.

- **conda** (`~/miniforge3`): `simfoundry`, `hunyuan`, `void`, `nerfstudio_simfoundry`, `3dgrut`.
- **Ollama 0.40.1**: 실행 파일은 `/home/inner/real2sim/tools/ollama/bin/ollama`(따로 받은 설치본, 876 MB)이고 모델은 `/data/models/ollama`(`qwen3-vl:8b-instruct`)입니다. `/home/inner/real2sim/tools`를 지우면 Ollama도 함께 지워집니다.
- **ComfyUI: 지금 이 머신에 없습니다.** 2026-10-16에 확인했을 때 설치 폴더와 Qwen-Image 모델 파일이 없었고 8189 포트도 비어 있었습니다. 재현 때는 127.0.0.1:8189에서 돌던 ComfyUI를 썼습니다. 이미지 모델을 부르는 단계(A 5·6단계, B 1단계)를 다시 돌리려면 다시 설치해야 합니다. `repro/run_local_pipeline.sh`는 시작할 때 ComfyUI 연결부터 확인하므로, 지금은 이 래퍼로 실행하면 바로 멈춥니다. 필요한 것은 `simfoundry/models/local_backends.py`의 기본값과 같습니다.
  - ComfyUI와 ComfyUI-GGUF(`UnetLoaderGGUF`) 노드
  - Edit-2511(기본): `qwen-image-edit-2511-Q3_K_M.gguf`, `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`, `qwen_2.5_vl_7b_fp8_scaled.safetensors`, `qwen_image_vae.safetensors`
  - 2.1-Turbo(`SIMFOUNDRY_IMAGE_BACKEND=qwen21`): `qwen-image-2.1-turbo-Q8_0.gguf`, `qwen3vl_8b_int8_convrot.safetensors`, `qwen_image_2.1_vae_bf16.safetensors`
- **Isaac Sim 5.1**: simfoundry 환경에 wheel로 설치했습니다. 이 회선은 큰 파일을 받다가 끊길 때가 있어서 [`fetch_isaac_wheels.sh`](fetch_isaac_wheels.sh)로 이어받기하며 받았습니다.
- **git**: `git clone`이 TLS 도중에 끊길 때가 있습니다. [`bin/git`](bin/git)이 clone만 다시 시도하고, `env.sh`가 이것을 PATH 앞에 넣습니다. 실행 중에 `rp.git_import`가 clone을 합니다.
- **hunyuan 환경 수정**: basicsr가 torchvision 0.22에 없는 모듈을 import합니다. [`fix_hunyuan_basicsr.sh`](fix_hunyuan_basicsr.sh)가 그 한 줄을 고칩니다. 환경을 다시 만들면 다시 실행해야 합니다.
- **Google Drive**: rclone 1.75.2(`~/.local/bin/rclone`)를 씁니다. 원격 이름은 `innerking`이고(`~/.config/rclone/rclone.conf`, scope `drive`), 루트는 Inner-King 폴더입니다. `repro/drive_sync.sh`는 `INNERINNER_DRIVE=innerking:innerinner`를 씁니다.

## 디스크와 정리 메모 (재현 당시, 2026-10-12)

재현 기록의 머신 쪽 메모입니다. 2026-10-16 이후 데이터와 가중치는 `/data`에 있습니다.

- 사용 공간: conda 환경 simfoundry 32 GB, hunyuan 13 GB, void 8.1 GB, nerfstudio 15 GB, 3dgrut 14 GB / `SimFoundry/`(deps, Data 포함) 13 GB / `tools/`(Ollama와 모델, SAM 3.1 변환본) 약 14 GB / HF 캐시 증가분 약 22 GB(Hunyuan3D-2.1, DA3) / VOID 가중치 41 GB.
- HF 토큰 저장 위치: `~/.cache/huggingface/token`, `SimFoundry/api_keys.txt`, `SimFoundry/scripts/installation/api_keys.txt`(권한 600, `api_keys.txt`는 .gitignore 대상). 채팅에 붙여 넣은 토큰이라 재발급(rotate)을 권합니다.
- 정리한 임시 파일: Isaac Sim wheel 사본(설치 후 삭제), ComfyUI `input/simfoundry_local` 업로드 이미지(삭제).
- 첫 설치 시도는 `PIP_NO_CACHE_DIR` 없이 시작해서 `~/.cache/pip`가 3.3 GB에서 3.9 GB로 늘었습니다.

## 2026-10-16 정리 전 위치

| 예전 위치 (`/home/inner/real2sim/…`) | 지금 |
|---|---|
| `REPRODUCTION.md` | `main`의 `repro/README.md`. 10장(디스크·정리 메모)만 이 파일로 옮김 |
| `run_local_pipeline.sh`, `rerun_v2.sh`, `run_auto_bg.sh` | `repro/` |
| `tools/*.py`, `tools/tasks/` | `repro/`, `repro/masked_bg/`, `repro/tasks/` |
| `tools/fruit_slicing/` | `repro/fruit_slicing/` |
| `tools/void-low-vram.patch` | `patches/void-model-low-vram.patch` (`install_void.sh`가 적용) |
| `tools/bin/git`, `tools/fetch_isaac_wheels.sh`, `tools/isaac_wheels*.txt` | `local/` |
| `tests/*.py`, `tests/q21/*.py` | `repro/diag/`, `repro/q21/` |
| `results_view/`, `tests/`의 이미지·영상 | Drive `innerinner/results/results_view/`, `results/diagnostics/` |
| `~/simfoundry_logs/` | Drive `innerinner/results/logs/` (로컬에도 계속 씀) |
| `SimFoundry/`의 커밋 안 된 auto_bg 변경 | `main` 커밋 |
