# SimFoundry (arXiv 2606.28276) 재현 기록

- 논문: *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation* (v4, 2026-08-05)
- 코드: [NVlabs/SimFoundry](https://github.com/NVlabs/SimFoundry) @ `9e34ebe` (2026-08-26)
- 재현 장비: i9-12900K, RAM 31 GiB, **RTX 3060 12 GB**. 논문은 RTX 3090 24 GB를 썼고, 저장소 최소 사양은 16 GB입니다.
- 이 폴더(`repro/`)에는 이 기록과 재현에 쓴 실행 래퍼(`run_local_pipeline.sh`, `start_ollama.sh`), 평가 스크립트(`eval_pose_fit.py`)가 있습니다. 아래 경로는 저장소 루트 기준입니다.

## 1. 재현 범위

| 논문 내용 | 재현 가능 여부 | 이유 |
|---|---|---|
| 동영상 → 시뮬레이션 장면 복원 (Pipeline A) | 재현 (로컬 모델로 대체) | 코드 공개 |
| Digital cousin·태스크 생성 (Pipeline B) | 재현 (로컬 모델로 대체) | 코드 공개 |
| OmniGibson 로딩·물리 스모크 테스트 (Pipeline C) | 재현 | 코드 공개 |
| 복원 소요 시간·물체 수 (Table L.4) | 비교 | 같은 종류의 장면으로 측정 |
| 정책 학습·sim-real 상관계수 0.911, 성공률 +17/21/40% (Fig. 4·5, Table 2, G.x) | 불가 | 데이터 생성·학습 코드 미공개(README "Not yet released"), 실제 DROID/YAM 로봇 필요 |
| 기하 정확도 Chamfer/F1 (Table L.2) | 불가 | YCB 장면의 준정답(quasi-GT) 스캔 미공개 |
| 자동 배경 복원 (Gaussian splat) | 생략 | 선택 기능이고 VOID 가중치가 게이트, 41 GB |

## 2. 원본과 다른 점 (대체·조정)

| 원본 | 이 재현 | 이유 |
|---|---|---|
| Gemini `gemini-3.1-pro-preview` / `2.5-pro` (텍스트·비전) | Ollama `qwen3-vl:8b-instruct` (Q4_K_M) | Gemini 키 없음 |
| Gemini `gemini-3-pro-image` (물체 제거, 업샘플, cousin 이미지) | ComfyUI Qwen-Image-Edit-2511 Q3_K_M + Lightning 4-step LoRA | 위와 같음 |
| `facebook/sam3` 체크포인트 | 로컬 SAM 3.1 가중치의 `detector.*` (이미지 모델) | 실행 당시 HF 게이트 승인 대기 중. 2026-10-10 승인 후 원본과 비교 (아래) |
| 박스 좌표 [ymin,xmin,ymax,xmax] (Gemini 규약) | Qwen의 [x1,y1,x2,y2] 출력을 shim에서 변환 | Qwen은 순서 지정 요청을 따르지 않음 (확인함) |
| 5단계 PDA 기하 깊이 백엔드 `da3` (DA3-giant fp32 6.8 GB) | `depth_pro` (설정 옵션 `s5_scene.pda_geometric_backend`) | SAM3+DA3+PDA+SBERT 상주량이 약 13 GB로 12 GB 초과 |
| 7단계 `s7_mesh.low_vram=false` | `true` + 텍스처 모델 CPU 대기·VAE slicing·베이킹 전 확산 모델 CPU 이동 (`mesh_generator.py` Hunyuan) | 형상 최대 6.5 GB, 텍스처 최대 10.9 GB로 측정됨. 원래 코드는 텍스처 모델 6.6 GB를 GPU에 상주시킴 |
| 5~8단계 스트리밍(병렬) | `--no-stream` (순차) | 12 GB에서 단계 동시 실행 불가 |
| 5단계 물체 제거 프롬프트 순서 `[mask, outline, bbox]` | `[outline, bbox]` | Qwen-Image-Edit는 초록으로 칠한 실루엣을 빈 공간이 아니라 같은 모양의 반투명 물체(유리잔, 반투명 가위)로 바꿈. 초록이 남지 않아 검사를 통과하고, 다음 반복에서 VLM이 그 '유령'을 다시 검출해 루프에 빠짐(18회 반복에서 중단). 설정 주석이 FLUX에 권하는 것과 같은 조치 |
| 원격 호출 중에도 GPU 모델 상주 | 로컬 백엔드 호출 동안 호출 프로세스의 CUDA 모듈을 CPU로 잠시 이동 (`gpu_released`) | ComfyUI·Ollama가 같은 GPU를 써야 함 |

환경 수정: hunyuan 환경의 basicsr가 torchvision 0.22에서 사라진 `functional_tensor`를 import해서, 해당 import 한 줄을 `functional`로 고쳤습니다(잘 알려진 호환성 문제).

SAM 3.1 가중치를 SAM3 이미지 모델에 넣었을 때 빠지는 키는 RoPE 버퍼(`freqs_cis`, 초기화 시 계산), 쓰이지 않는 최저 해상도 neck(`convs.3`, scalp=1로 버려짐), `text_projection`입니다. 시험 결과 그릇 0.97, 컵 0.88, 테이블 0.90 점수에 박스도 정확했습니다.

**원본 SAM3와 비교 (2026-10-10 승인 후)**: 같은 시험 이미지에서 원본 `facebook/sam3`(빠진 키 없이 로드)와 SAM 3.1 대체본의 결과가 거의 같았습니다. 그릇 0.97/0.97, 컵 0.89/0.88, 테이블 0.90/0.90, 박스 차이는 4 px 이내이고 펜·마커는 둘 다 미검출입니다. 따라서 위 두 장면의 결과는 SAM 대체의 영향을 거의 받지 않았다고 봅니다. 이후 실행은 `repro/run_local_pipeline.sh`의 기본값으로 원본 SAM3를 씁니다.

구현은 `simfoundry/models/local_backends.py` 한 파일에 모았습니다. 스테이지 코드는 그대로 두고 `genai.Client`만 바꿔 끼웁니다(`SIMFOUNDRY_LOCAL_BACKEND=1`일 때만). 실행은 `repro/run_local_pipeline.sh`로 합니다.

## 3. 결과: Pipeline A (동영상 → 시뮬레이션 장면)

입력: 저장소 예제 동영상 `docs/assets/example_videos/PutMarkerInCup.mp4` (1920×1080, 14.9초). 출력: `Data/put_marker_in_cup/`.

### 3.1 단계별 결과

| 단계 | 결과 | 소요 시간 |
|---|---|---|
| 1b 동영상 처리 | 15프레임 추출 | 8 s |
| 2 깊이 (DA3-giant) | 15프레임 메트릭 깊이 | 16 s |
| 3 바닥면 분할·프레임 선택 | 책상(desk) 평면. VLM 프레임 선택이 토큰 한도에 걸려 휴리스틱 결과(프레임 5) 사용 | 221 s * |
| 4 좌표계 통일 | | 3 s |
| 5 장면 분해 | 3개 물체: 빨간 마커, 흰 종이컵, 파란 가위. 반복 제거 후 VLM이 빈 결과 3회 → 종료 | 338 s ** |
| 6 물체 이미지 업샘플 | 스튜디오 사진 3장과 배경 제거본 | 306 s |
| 7 메시 생성 (Hunyuan3D-2.1) | 3개 모두 형상과 PBR 텍스처 | 655 s *** |
| 8 포즈 정합 (FoundationPose) | 3개 | 108 s |
| 10~12 장면 컴파일·sim-ready·물리 안정화 | 질량: 가위 50 g, 마커 36 g, 컵 70 g. 마찰: 0.3/0.5/0.5 | 66 s |
| 13 USD 변환 | | 163 s |
| 14 OmniGibson 장면 | `s14_og/reconstructed_og_scene.json` | 63 s |
| **합계** | | **약 1,947 s ≈ 32.5분** (물체 3개, RTX 3060) |

\* 수정 전 측정값입니다. 이 VLM 호출이 반복 루프로 8192 토큰을 생성하는 데 196 s가 걸렸고, 이후 최대 토큰을 4096으로 줄이고 잘리면 한 번 샘플링으로 재시도하도록 고쳤습니다.
\** 업샘플·첫 검출 VLM 응답은 앞선 실행의 캐시를 재사용했습니다(약 1.5분 절약).
\*** 7단계 도중 마커 텍스처 베이킹이 OOM으로 실패해 수정 후 그 물체만 다시 돌렸습니다. 매니페스트 시각으로 합산했습니다(컵 252 s, 가위 185 s, 마커 218 s).

논문 Table L.4: 물체 2개 장면 **19.98분** (RTX 3090 24 GB, Gemini API, 5~8단계 스트리밍). 논문 FAQ: 물체당 약 5분.
이 재현: 물체당 **약 10.8분**. 12 GB에 맞추려고 단계를 순차로 돌리고, 메시 생성에 CPU 오프로드를 쓰고, 생성 모델을 로컬에서 돌린 결과입니다. 수치 자체보다는 *같은 자릿수에서 돌아간다*는 것이 재현 포인트입니다.

### 3.2 기하 품질

물체 크기 (metadata `bbox_size`, OmniGibson 단위 m):

| 물체 | 복원 크기 (cm) | 일반적인 실물 크기 (참고) |
|---|---|---|
| 흰 종이컵 | 9.5 × 9.5 × 11.0 | 12 oz 종이컵: 지름 약 9 cm, 높이 약 11 cm |
| 빨간 마커 | 13.1 × 2.1 × 2.0 | 화이트보드 마커: 길이 약 13~14 cm, 지름 약 2 cm |
| 파란 가위 | 21.9 × 9.4 × 2.1 | 8인치 가위: 길이 약 20 cm |

포즈 정합 잔차 (`repro/eval_pose_fit.py`): 관측 점구름(깊이 기반) → 포즈를 적용한 메시 표면 거리

| 물체 | 관측 점 수 | 평균 | 중앙값 | 1 cm 이내 | 5 mm 이내 |
|---|---|---|---|---|---|
| 종이컵 | 674 | 2.04 mm | 0.75 mm | 96.9% | 87.5% |
| 가위 | 783 | 2.02 mm | 1.76 mm | 100.0% | 94.8% |
| 마커 | 275 | 2.86 mm | 2.46 mm | 96.7% | 90.5% |

이 값은 정합에 사용한 관측과 비교한 단방향 잔차이므로, 논문 Table L.2(준정답 스캔 대비 양방향 Chamfer 4.2 mm / F1 0.92, Easy)와 직접 비교할 수는 없습니다. 메시 형상과 메트릭 스케일·포즈가 관측과 일관된다는 것만 보여 줍니다.

저자가 공개한 `droid_desk_put_marker_in_cup` 장면과의 비교: 저자 장면은 손잡이 있는 `cream_mug`(9.4×7.3×8.8 cm)와 마커(11.4 cm) 두 개뿐이고 두 물체 거리가 0.463 m입니다. 예제 동영상에는 종이컵·마커·가위가 있고 우리 복원의 컵-마커 거리는 0.238 m입니다. 물체 구성과 배치가 달라서 같은 촬영본이 아닌 것으로 판단했고, 정답으로 쓰지 않았습니다. 저장소 문서에 따르면 공개 장면은 B와 C 사이에 사람이 편집합니다.

## 4. 결과: Pipeline C (OmniGibson 스모크 테스트)

`C_application --mode smoke-random`: 복원 장면을 OmniGibson(Isaac Sim 5.1)에 불러와 Franka가 무작위 동작 20스텝을 수행했습니다. **통과**(48 s). 영상: `Data/put_marker_in_cup/application_smoke/random_action_smoke.mp4`. 세 물체가 텍스처와 함께 렌더링되고 물리 시뮬레이션 중에 오류가 없었습니다.

## 5. 결과: Pipeline B (digital cousin과 태스크 제안)

실행 설정: README 예시와 같이 `prompt_cousin_structured.max_objects=2`, `max_generated_images_per_object=1`, `cousin_generation.low_vram=true`.

| 단계 | 결과 | 소요 시간 |
|---|---|---|
| 1 cousin 변형 제안·이미지 생성 | 컵: 부품 [Main Body, Rim], 첫 변형 "같은 바닥 지름에서 높이를 키운 더 긴 컵". 가위: 부품 [Handle, Blade], "손가락 고리 확대" → 손잡이 형태·색이 바뀐 가위 | 292 s |
| 2 cousin 조합 | | 4 s |
| 3 cousin 메시 (Hunyuan3D-2.1) | 2개 | 430 s |
| 4 cousin sim-ready | 정면 정렬·질량·마찰 (아래 버그 수정 후) | 85 s |
| 5 cousin USD 변환 | | 30 s |
| 6 장면 변형 샘플링 | 2개 생성, 2/2 물리적으로 유효 (`s14_og/auto_generation/scene_00{0,1}`) | 80 s |
| 7 태스크 제안 | `place_scissors_in_cup` (InsideAABB), `pick_up_marker` (Lifted ≥ 5 cm). OmniGibson `PickPlaceTask` YAML | 11 s |
| **합계** | | **약 15.5분** |

이 단계에서 찾은 문제 두 가지:
- **원본 코드 버그 (B4)**: `trimesh.load(...glb)`가 trimesh 4.x에서 `Scene`을 반환해서 정면 정렬 렌더가 실패하고, 이어서 존재하지 않는 폴더에 `orientation.json`을 쓰다가 크래시가 납니다. A의 8단계처럼 `force="mesh"`를 넣어 고쳤습니다(`4_make_cousins_sim_ready.py:146`). requirements에서 `trimesh`는 버전이 고정되지 않았습니다(`#==3.23.5` 주석 처리).
- **로컬 이미지 모델 특성**: "transparent background"를 요청하면 Qwen-Image-Edit가 체커보드 무늬와 소품을 그려 넣습니다. rembg 후처리가 이를 깨끗이 제거해서 이번 결과에는 영향이 없었지만, 이후 실행부터는 shim이 이 표현을 "plain pure white background"로 바꿔 보냅니다.

## 6. 두 번째 장면: PutCupInBowl (모든 수정 반영, 캐시 없이 한 번에 실행)

입력: `docs/assets/example_videos/PutCupInBowl.mp4`. 출력: `Data/put_cup_in_bowl/`. 1b~14단계가 **한 번에 성공**했습니다.

| 단계 | 1b | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | **합계** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 시간 (s) | 8 | 16 | 47 | 3 | 461 | 219 | 647 | 112 | 3 | 79 | 3 | 48 | 49 | **1,695 (28.2분)** |

- 검출·분리한 물체: 주황 그릇, 검은 펜, 흰 컵 (3개). 빈 검출 3회로 정상 종료. 유령 물체 없음.
- 복원 크기: 그릇 25.8×25.9×13.6 cm, 펜 13.0×0.9×0.6 cm, 컵 9.3×9.3×10.6 cm. 그릇은 원본 프레임에서 컵 폭의 약 3배로 보이므로 크기가 그럴듯합니다.
- 정합 잔차 (관측 → 메시): 컵 평균 2.5 mm (1 cm 이내 96%), 펜 6.0 mm (73%), 그릇 12.7 mm (49%). 그릇은 외곽 범위가 관측과 1~3 cm 이내로 맞습니다(폭 25.5 vs 25.7 cm). 1 cm를 넘는 점은 대부분 앞 테두리보다 깊은 안쪽 면(관측 z 평균 0.73 m vs 0.68 m)에 있습니다. 광택 있는 오목면 안쪽은 단안 깊이 추정이 어려운 영역입니다.
- VLM 질량 추정: 그릇 0.58 kg, 컵 0.15 kg, 펜 0.04 kg. 컵과 펜은 실물보다 무겁게 추정됐습니다(Gemini 대신 8B 로컬 VLM을 쓴 영향일 수 있음).
- Pipeline C 스모크 테스트: **통과** (45 s).

## 7. 논문 대비 요약

| 항목 | 논문 | 이 재현 | 판정 |
|---|---|---|---|
| 동영상 하나로 sim-ready 장면 자동 생성 | 수동 주석 없이 1시간 이내 | 두 장면 모두 사람 개입 없이 완료(설정 오버라이드만). 28~32분 | 재현됨 |
| 복원 시간 (Table L.4, 물체당 약 5분, RTX 3090) | 2개 물체 19.98분 | 3개 물체 28.2분(캐시 없음) / 32.5분, 물체당 9.4~10.8분 (RTX 3060 12 GB, 순차 실행, 로컬 모델) | 같은 자릿수. 약 2배 느림 (예상 범위) |
| 물체별 메시·포즈·물리 파라미터 | Hunyuan3D-2.1, FoundationPose, VLM 질량·마찰 | 같은 모듈 사용 (VLM만 로컬). 크기가 실물 크기대와 맞음 | 재현됨 |
| OmniGibson 로딩·시뮬레이션 | | 두 장면 모두 스모크 테스트 통과 | 재현됨 |
| Digital cousin (형상·위상·외관 변형) | Gemini 텍스트 + 이미지 | 변형 제안과 이미지·메시 생성 성공(더 긴 컵, 손잡이가 바뀐 가위). 장면 변형 2/2 유효 | 재현됨 (로컬 모델) |
| 태스크 제안 | | 2개 PickPlaceTask 생성 | 재현됨 |
| 기하 정확도 (Table L.2) | Chamfer 4.2 mm, F1 0.92 (Easy) | 준정답 스캔 미공개라 불가. 대신 관측 대비 잔차를 보고 (평균 2~6 mm, 그릇 12.7 mm) | 직접 비교 불가 |
| 정책 학습·sim-real 상관 (0.911)·성공률 개선 (+17/21/40%) | | 코드 미공개, 실제 로봇 필요 | 재현 불가 |

**결론**: 공개된 범위(Pipeline A·B·C)는 Gemini 없이 로컬 모델로 바꾸고 12 GB GPU에 맞게 조정한 상태에서, 저장소 예제 동영상 두 개에 대해 끝까지 재현했습니다. 논문의 핵심 정량 결과(sim-to-real 정책 성능)는 데이터 생성·학습 코드가 공개되지 않았고 실제 로봇이 필요해서 재현할 수 없습니다.

## 8. 다시 실행하는 법

```bash
# Pipeline A (장면 복원)
repro/run_local_pipeline.sh A_reconstruction --scene-name NAME --video-fpath /abs/path/VIDEO.mp4 \
  --env-da3 simfoundry --no-stream -- \
  s7_mesh.low_vram=true s5_scene.pda_geometric_backend=depth_pro \
  's5_scene.obj_removal_prompt_order=[outline,bbox]'
# Pipeline B (cousin과 태스크)
repro/run_local_pipeline.sh B_augmentation --scene-name NAME -- \
  prompt_cousin_structured.max_objects=2 prompt_cousin_structured.max_generated_images_per_object=1 \
  cousin_generation.low_vram=true
# Pipeline C (OmniGibson 스모크 테스트)
repro/run_local_pipeline.sh C_application --scene-name NAME --mode smoke-random
# 평가
~/miniforge3/envs/simfoundry/bin/python repro/eval_pose_fit.py Data/NAME
```

전제 조건: ComfyUI(127.0.0.1:8189)가 실행 중이어야 합니다. Ollama는 래퍼가 필요할 때 띄웁니다. `--video-fpath`는 절대 경로로 줍니다(스테이지가 다른 작업 디렉터리에서 실행됨). Ollama 실행 파일이나 모델 저장소가 기본 위치가 아니면 `OLLAMA_BIN`, `OLLAMA_MODELS`로 지정합니다. 체크아웃에 `local/env.sh`가 있으면(머신별 브랜치, 예: `hanbin5/local`) 래퍼가 먼저 읽어서 이런 값을 채웁니다.

## 9. 수정한 파일 목록

| 파일 | 변경 |
|---|---|
| `simfoundry/models/local_backends.py` (신규) | Gemini 클라이언트 대체: Ollama/ComfyUI 라우팅, 좌표 변환, GPU 오프로드, VRAM 반환 대기, 반복 루프 재시도, 반복 요청 seed 변경, 투명 배경 프롬프트 변환, SAM3 체크포인트 경로 |
| `simfoundry/models/vlm.py` | `SIMFOUNDRY_LOCAL_BACKEND=1`일 때 로컬 클라이언트를 반환하고 인증을 건너뜀 (두 곳) |
| `simfoundry/models/sam_v3.py`, `sam_v3_gmask.py` | `SIMFOUNDRY_SAM3_CHECKPOINT`가 있으면 로컬 체크포인트 사용 |
| `simfoundry/models/mesh_generator.py` (Hunyuan) | `low_vram`: 텍스처 모델 CPU 대기, VAE slicing, 베이킹 전 확산 모델 CPU 이동 |
| `scripts/pipeline/B_augmentation/stages/4_make_cousins_sim_ready.py:146` | `trimesh.load(..., force="mesh")` (원본 버그 수정) |
| `scripts/pipeline/B_augmentation/stages/1_prompt_object_cousins.py` (`parse_reasonableness_response`) | JSON 형식 답변도 파싱되도록 따옴표 허용 (v2) |
| `simfoundry/models/local_backends.py` (v2 추가분) | Qwen-Image-2.1(-Turbo) 백엔드 `SIMFOUNDRY_IMAGE_BACKEND=qwen21`, 불투명 RGBA 출력 → RGB |
| hunyuan env `basicsr/data/degradations.py` | `functional_tensor` → `functional` |

## 10. 추가 실험: 이미지 백엔드 비교 (2026-10-10)

Qwen-Image-2.1은 텍스트→이미지 전용이 아니라 생성·편집 통합 모델입니다. 칠한 표시·마스크로 국소 편집을 지정할 수 있고 RGBA를 직접 생성합니다. 그래서 파이프라인이 실제로 보내는 입력·프롬프트 6개로 세 모델을 비교했습니다 (비교 스크립트와 비교 이미지는 이 포크에 넣지 않았습니다).

| 항목 | Qwen-Image-Edit-2511 (Q3_K_M, Lightning 4스텝) | Qwen-Image-2.1-Turbo (Q8_0, 8스텝) | Qwen-Image-2.1 기본 (Q8_0, 25스텝) |
|---|---|---|---|
| 5단계 mask 방식 제거 (원본 기본값) | 실패: 같은 모양의 유리잔 '유령' 생성 | 성공 | 성공 |
| 5단계 outline 방식 제거 (컵, 가위) | 성공 | 성공 | 성공 |
| 주석 영역 밖 평균 변화 (0-255, 낮을수록 보존 잘 됨; mask컵/outline컵/outline가위) | **1.7 / 6.2 / 2.1** | 6.1 / 9.9 / 6.0 | 4.1 / 7.7 / 3.6 |
| 6단계 업샘플 (마커, 가위) | 프롬프트의 45° 시점은 따르나 라벨 글자를 지어냄 | 원래 자세와 라벨에 충실 | Turbo와 거의 같음 |
| B 단계 cousin "더 긴 컵" (원본 높이/폭 1.35) | 1.34 (변형 미반영), 흰 배경 | **1.60**, 진짜 투명 배경 (투명 72%) | 1.54, 진짜 투명 배경 |
| 1회 평균 시간 (RTX 3060) | 58.6 s | **51.7 s** | 115.1 s |

결론: 이 파이프라인에는 **Qwen-Image-2.1-Turbo가 더 적합**합니다. Edit-2511 때문에 우회해야 했던 두 문제(mask 방식의 유령, "투명 배경"을 체커보드로 그림)가 2.1에서는 생기지 않아 원본 설정 `[mask, outline, bbox]`를 그대로 쓸 수 있습니다. 약점은 편집 영역 밖의 색조 변화가 조금 더 크다는 점입니다(5단계가 물체 주변 마스크 안에서만 합성하므로 영향은 작음). 주석 없는 자유 프롬프트("흰 컵을 지워라")에서는 컵 아래 그릇까지 지운 사례가 한 번 있었습니다.

사용법: `SIMFOUNDRY_IMAGE_BACKEND=qwen21`. 기본 모델은 `SIMFOUNDRY_Q21_UNET=qwen-image-2.1-Q8_0.gguf SIMFOUNDRY_Q21_STEPS=25`로 지정합니다. 2.1은 항상 RGBA로 디코딩하므로 shim이 거의 불투명한 출력은 RGB로 바꾸고(5단계는 3채널을 전제함), 실제 투명 출력만 RGBA로 남깁니다. 위 3~6장의 결과는 모두 Edit-2511로 만든 것입니다.

## 11. v2 재실행: 원본에 가장 가까운 구성 (2026-10-10)

구성: 이미지 모델 **Qwen-Image-2.1-Turbo** (`SIMFOUNDRY_IMAGE_BACKEND=qwen21`), 물체 제거는 **원본 기본 순서 `[mask, outline, bbox]`**(오버라이드 없음), **원본 `facebook/sam3`**. 12 GB에 맞추려고 남긴 조정은 `depth_pro` PDA 백엔드, Hunyuan `low_vram`, 순차 실행입니다. 응답 캐시는 별도 폴더를 썼습니다(캐시 키에 백엔드가 없어 v1 이미지가 재사용될 수 있음). 출력: `Data/put_marker_in_cup_v2`, `Data/put_cup_in_bowl_v2`. 실행 명령은 이 장 끝에 있습니다.

| | PutMarkerInCup v1 → **v2** | PutCupInBowl v1 → **v2** |
|---|---|---|
| A 실행 | 수정하며 여러 번 → **한 번에 성공** | 한 번에 → **한 번에 성공** |
| A 소요 시간 | 32.5분* → **28.5분** | 28.2분 → **29.1분** |
| VLM 프레임 선택 | 실패(휴리스틱 대체) → **VLM 선택 성공** | 성공 → 성공 |
| 물체 제거 방식 | outline 강제 → **mask 성공(컵), mask 3회 무반응 후 outline(가위)** | outline 강제 → **mask 2개, outline 1개** |
| 유령 물체 루프 | (mask 사용 시 발생) → **없음** | 없음 → 없음 |
| 크기 (cm) | 마커 13.1 / 컵 11.0 높이 / 가위 21.9 → **13.7 / 11.2 / 20.0** | 그릇 25.8 / 펜 13.0 / 컵 10.6 → **25.7 / 12.8 / 11.0** |
| 정합 잔차 평균 / 1 cm 이내 | 2.31 mm / 97.9% → **2.55 mm / 98.5%** | 7.04 mm / 72.6% → **5.91 mm / 81.7%** (펜 5.95 → 2.85 mm, 73 → 100%) |
| C 스모크 테스트 | 통과 → **통과** | 통과 → **통과** |

\* v1 첫 장면은 중간에 수정하며 나눠 돌렸고 일부 캐시를 재사용했습니다.

**mask 방식에서 2.1-Turbo의 실제 동작**: 컵·그릇처럼 큰 물체는 mask 첫 시도에서 깨끗하게 지웠습니다. 가는 가위는 초록 실루엣을 **그대로 둔 채(편집 무반응)** 돌려줬습니다. 그래서 단계의 "초록 잔여" 검사가 3회 모두 거부했고, 원래 설계된 대체 순서대로 outline으로 넘어가 성공했습니다. Edit-2511처럼 검사를 통과하는 '유령'을 만들지 않으니, 원본 파이프라인의 안전장치가 그대로 작동합니다.

**Pipeline B v2** (PutMarkerInCup): 1단계 240 s(첫 시도), 3단계 428 s, 4~7단계 204 s.
- cousin 이미지가 처음부터 **실제 투명 배경(RGBA)**으로 생성됐습니다(체커보드 없음).
- 컵 "더 길고 좁게"는 반영됐고, 가위 "손가락 고리 확대"는 거의 반영되지 않았습니다.
- 장면 변형 2/2 유효.
- 제안된 태스크: **`pick_marker_and_place_in_cup`**(동영상의 실제 과제와 일치), `pick_scissors_and_place_above_cup`.
- 수정 1건: 판정 단계에서 로컬 VLM이 맞는 답(`yes`)을 JSON 형식(`"RealWorldReasonable": "yes"`)으로 내자 파서가 형식 오류로 3회 모두 거부했습니다. `1_prompt_object_cousins.py`의 `parse_reasonableness_response` 정규식이 따옴표를 선택적으로 허용하도록 고쳤습니다(일반 텍스트 형식도 그대로 파싱됨).

**결론**: 2.1-Turbo와 원본 기본 설정으로 두 장면 모두 사람 개입 없이 한 번에 끝났습니다. 품질은 v1과 같거나 나았습니다(펜 정합 크게 개선, 가위 크기가 실물에 더 가까움). 다만 VLM이 추정하는 질량은 여전히 실행마다 흔들립니다(종이컵 70 g ↔ 150 g).

실행 명령 (저장소 루트에서):

```bash
export SIMFOUNDRY_IMAGE_BACKEND=qwen21
# 응답 캐시를 v1과 분리합니다(캐시 키에 백엔드가 들어가지 않음).
CACHE=(--cache-mode --model-cache-dir .cache/simfoundry/model_calls_qwen21)
V="$PWD/docs/assets/example_videos"
for pair in "put_marker_in_cup_v2:PutMarkerInCup" "put_cup_in_bowl_v2:PutCupInBowl"; do
  scene=${pair%%:*}; video=${pair##*:}
  repro/run_local_pipeline.sh A_reconstruction --scene-name "$scene" --video-fpath "$V/$video.mp4" \
    --env-da3 simfoundry --no-stream "${CACHE[@]}" -- \
    s7_mesh.low_vram=true s5_scene.pda_geometric_backend=depth_pro
  repro/run_local_pipeline.sh C_application --scene-name "$scene" --mode smoke-random
done
repro/run_local_pipeline.sh B_augmentation --scene-name put_marker_in_cup_v2 "${CACHE[@]}" -- \
  prompt_cousin_structured.max_objects=2 prompt_cousin_structured.max_generated_images_per_object=1 \
  cousin_generation.low_vram=true
```
