# SimFoundry (arXiv 2606.28276) 재현 기록

- 논문: *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation* (v4, 2026-08-05)
- 코드: [NVlabs/SimFoundry](https://github.com/NVlabs/SimFoundry) @ `9e34ebe` (2026-08-26)
- 재현 장비: i9-12900K, RAM 31 GiB, **RTX 3060 12 GB**. 논문은 RTX 3090 24 GB를 썼고, 저장소 최소 사양은 16 GB입니다. 12 GB 때문에 바꾸거나 못 한 실행은 Colab G4 런타임(**RTX PRO 6000 Blackwell 96 GB**)에서 저장소 기본값대로 다시 돌렸습니다(17장).
- 이 폴더(`repro/`)에는 이 기록과 재현에 쓴 실행 래퍼와 스크립트가 모두 있습니다(목록은 15장). 아래 경로는 저장소 루트 기준입니다.
- 결과 영상·이미지·로그, 장면 데이터(`Data/<scene>`), 체크포인트는 git에 넣지 않고 Google Drive [`Inner-King/innerinner/`](https://drive.google.com/drive/folders/1lmzFWqMATc52zW7gyiog0YZM3ewKDUzY)에 둡니다(14장). 이 기록의 `results_view/…`, `diagnostics/…`, `logs/…` 경로는 그 폴더의 `results/` 아래입니다.

## 1. 재현 범위

| 논문 내용 | 재현 가능 여부 | 이유 |
|---|---|---|
| 동영상 → 시뮬레이션 장면 복원 (Pipeline A) | 재현 (로컬 모델로 대체) | 코드 공개 |
| Digital cousin·태스크 생성 (Pipeline B) | 재현 (로컬 모델로 대체) | 코드 공개 |
| OmniGibson 로딩·물리 스모크 테스트 (Pipeline C) | 재현 | 코드 공개 |
| 복원 소요 시간·물체 수 (Table L.4) | 비교 | 같은 종류의 장면으로 측정. 3060(7장)과 96 GB G4(17장) |
| 정책 학습·sim-real 상관계수 0.911, 성공률 +17/21/40% (Fig. 4·5, Table 2, G.x) | 불가 (학습) / 부분 가능 (평가) | README는 "Not yet released"라고 하지만, 저장소에 시연 생성·정책 평가 코드(π0/π0.5/GR00T/DreamZero 클라이언트)는 들어 있음. 빠진 것은 학습 설정·학습된 체크포인트·실제 로봇 데이터. Table G.1의 시뮬 쪽은 재현: 저자 공개 장면 5개에서 공개 π0·π0.5-DROID(18장), 우리 복원 장면과 GR00T N1.6·DreamZero(19장) |
| 기하 정확도 Chamfer/F1 (Table L.2) | 불가 | YCB 장면의 준정답(quasi-GT) 스캔 미공개 |
| 자동 배경 복원 (Gaussian splat, App. E.5.1) | 부분 재현 | 저장소 auto_bg 7단계가 끝까지 돌고 촬영 시점에서는 맞지만, 로봇 카메라 시점에서는 배경이 무너짐(17.4). 재학습과 후처리로 책상과 물체는 보이게 됐지만, 영상에 없던 영역은 하늘과 파편으로 남음(17.7). 손목 카메라에서는 기본 배경이 깨지고, 17.7의 배경은 책상을 내려다볼 때만 알아볼 만함(30.2). 12 GB용 수정과 로컬 VLM 보완은 13장(PutMarkerInCup), 저장소 기본값(400프레임 등)으로 두 장면은 17장 |

## 2. 원본과 다른 점 (대체·조정)

| 원본 | 이 재현 | 이유 |
|---|---|---|
| Gemini `gemini-3.1-pro-preview` / `2.5-pro` (텍스트·비전) | Ollama `qwen3-vl:8b-instruct` (Q4_K_M). 2026-10-16부터 Pipeline A는 `qwen3.5:9b`(Q4_K_M, thinking 끔, 16장) | Gemini 키 없음 |
| Gemini `gemini-3-pro-image` (물체 제거, 업샘플, cousin 이미지) | ComfyUI Qwen-Image-Edit-2511 Q3_K_M + Lightning 4-step LoRA | 위와 같음 |
| `facebook/sam3` 체크포인트 | 로컬 SAM 3.1 가중치의 `detector.*` (이미지 모델) | 실행 당시 HF 게이트 승인 대기 중. 2026-10-10 승인 후 원본과 비교 (아래) |
| 박스 좌표 [ymin,xmin,ymax,xmax] (Gemini 규약) | Qwen의 [x1,y1,x2,y2] 출력을 shim에서 변환 | Qwen은 순서 지정 요청을 따르지 않음 (확인함). Qwen3.5도 thinking을 끄면 같고, 켜면 요청한 순서를 따름 (16장) |
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

저자가 공개한 `droid_desk_put_marker_in_cup` 장면과의 비교 (`repro/compare_scenes.py`): 저자 장면은 손잡이 있는 `cream_mug`(9.4×7.3×8.8 cm)와 마커(11.4 cm) 두 개뿐이고 두 물체 거리가 0.463 m입니다. 예제 동영상에는 종이컵·마커·가위가 있고 우리 복원의 컵-마커 거리는 0.238 m입니다. 물체 구성과 배치가 달라서 같은 촬영본이 아닌 것으로 판단했고, 정답으로 쓰지 않았습니다. 저장소 문서에 따르면 공개 장면은 B와 C 사이에 사람이 편집합니다.

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
| 복원 시간 (Table L.4, 물체당 약 5분, RTX 3090) | 2개 물체 19.98분 | 3개 물체 28.2분(캐시 없음) / 32.5분, 물체당 9.4~10.8분 (RTX 3060 12 GB, 순차 실행, 로컬 모델). **96 GB G4에서 저장소 기본값(스트리밍)으로 8.7~9.1분, 물체당 약 3분** (17장) | 같은 자릿수. 3060은 약 2배 느림, G4는 논문 수치보다 빠름 |
| 물체별 메시·포즈·물리 파라미터 | Hunyuan3D-2.1, FoundationPose, VLM 질량·마찰 (정확도 평가 없음) | 같은 모듈 사용 (VLM만 로컬). 크기가 실물 크기대와 맞음. 질량은 YCB 28개 실측 대비 2배 안 비율이 원래 문구 43% → 지금 11단계 82%(Qwen3.5, 29장). 마찰은 VLM 답이 재질과 거의 무관하고 실측 자료가 없어, 시뮬레이터에 추정치가 그대로 들어가게만 함(29.5) | 재현됨, 질량 개선 |
| OmniGibson 로딩·시뮬레이션 | | 두 장면 모두 스모크 테스트 통과 | 재현됨 |
| Digital cousin (형상·위상·외관 변형) | Gemini 텍스트 + 이미지 | 변형 제안과 이미지·메시 생성 성공(더 긴 컵, 손잡이가 바뀐 가위). 장면 변형 2/2 유효 | 재현됨 (로컬 모델) |
| 태스크 제안 | | 2개 PickPlaceTask 생성 | 재현됨 |
| 기하 정확도 (Table L.2) | Chamfer 4.2 mm, F1 0.92 (Easy) | 준정답 스캔 미공개라 불가. 대신 관측 대비 잔차를 보고 (평균 2~6 mm, 그릇 12.7 mm) | 직접 비교 불가 |
| Table G.1 시뮬 성공률 (미세조정 없는 π0 / π0.5, 저자 장면) | Marker in Cup 40/88%, Serve Fruits 4/80% | 16/52%, 0/88% (18장, 25회씩) | Serve Fruits는 맞음. Marker in Cup은 두 정책 모두 낮음(질량·카메라·지시문, 톤매퍼, 배경 스캔의 로봇, 팔의 추적 지연, 잡기 방식, 손가락 속도 제한을 바꿔도 유의한 차이 없음: 18, 20.2, 23, 26, 28장). 손목 카메라 렌즈와 그리퍼 상태 입력도 맞음. 저자 장면은 공개되지 않은 OmniGibson 빌드로 저장됨(30.1, 30.5) |
| 과제별 sim-real 상관 (Table G.1, 제로샷 과제) | r 0.995 (Marker in Cup), 0.960 (Serve Fruits), 0.907 (Cup in Bowl). 정책 5개 | 정책 3개(π0, π0.5, GR00T N1.6)의 우리 시뮬 대 논문 실제: **0.999** (우리 복원 Marker in Cup), **0.993** (Serve Fruits), −0.14 / −0.48 (우리 복원 Cup in Bowl, 배치 두 가지) (19장). DreamZero-DROID를 넷째 정책으로 더하면 0.330 (우리 복원 Marker in Cup), 0.993 (Serve Fruits), −0.501 (우리 복원 Cup in Bowl) (19.4). 질량을 실물에 가깝게 고친 장면: 0.984 (Marker in Cup), 0.45 (Cup in Bowl) (21장). 우리가 복원한 Serve Fruits: 0.824, MMRV 0, 방해물인 주황 접시만 저자 것으로 바꾸면 **0.994** (22장). 저장소 태스크 설정의 배치로 뽑은 Cup in Bowl: 0.14 (25장), 여기에 physical 잡기: 0.19 (27장). 지금까지의 수정(11단계 새 질량, 손가락 제한, physical 잡기)을 모두 넣은 Cup in Bowl: 0.726, MMRV 0.133 (30.4, π0·GR00T 100회씩), 같은 수정의 우리 Marker in Cup: **0.991**, MMRV 0 (30.6), 같은 수정의 우리 Serve Fruits: **0.997**, MMRV 0 (30.8). 다만 수정을 모두 넣은 장면에서는 π0.5의 성공률이 논문 시뮬보다 크게 낮음(58%, 42%) | 정책 3개로는 3개 중 2개 재현. DreamZero는 우리 장면 두 과제에서 논문보다 크게 낮아(0%, 16%) 정책 4개로는 Serve Fruits만 맞음, 원인 미확인(19.4). Cup in Bowl은 저자 장면이 공개되지 않아 직접 비교 불가. 질량 오차도, 저장소 설정의 배치도, 잡기 방식도 따로는 원인이 아님(21, 25, 27장). 수정을 모두 넣어도 우리 시뮬에서는 π0와 GR00T가 같아(100회씩 72%, 74%) 실제의 차이(20%p)를 재현하지 못함. GR00T가 논문 시뮬(40%)보다 높음(30.4). 우리 Serve Fruits는 순위가 맞고 GR00T만 높았는데, 원인은 방해물 하나의 복원(22.4) |
| 정책 학습·성공률 개선 (+17/21/40%), 미세조정 과제의 상관 | | 학습 코드·미세조정 체크포인트 미공개, 실제 로봇 필요 | 재현 불가 |

**결론**: 공개된 범위(Pipeline A·B·C)는 Gemini 없이 로컬 모델로 바꾸고 12 GB GPU에 맞게 조정한 상태에서, 저장소 예제 동영상 두 개에 대해 끝까지 재현했습니다. 12 GB 조정을 모두 뺀 저장소 기본값으로도 96 GB GPU에서 같은 결과가 나왔습니다(17장). 자동 3DGS 배경은 촬영 시점 근처에서만 맞고 로봇 카메라 시점에서는 쓸 수 없습니다(17.4). 손목 카메라에서도 기본 배경은 깨지고, 재학습한 배경(17.7)은 책상을 내려다볼 때만 알아볼 만합니다(30.2). 논문의 핵심 정량 결과 가운데 "시뮬 평가가 실제 성능을 예측한다"는 주장은, 공개 체크포인트로 평가할 수 있는 제로샷 과제에서 부분적으로 재현했습니다(19장). 우리가 예제 영상에서 복원한 Marker in Cup 장면은 세 정책의 순위와 성공률이 논문의 실제 로봇 결과와 r 0.999로 맞았고, 저자 장면의 Serve Fruits는 0.993이었습니다. 네 번째 정책 DreamZero-DROID는 Serve Fruits에서는 논문과 맞았지만(16%, 논문 시뮬 12%), 우리 장면 두 과제에서는 0%와 16%로 논문 시뮬(80%, 92%)보다 크게 낮았습니다. 그래서 정책 넷으로는 Marker in Cup이 0.330으로 내려갑니다(19.4). Cup in Bowl은 어느 배치에서도, physical 잡기로도 맞지 않았습니다(25, 27장). 지금까지의 수정을 모두 넣어도 맞지 않았습니다. 처음 50회씩에서는 순위가 맞았지만, 100회로 늘리자 π0(72%)와 GR00T(74%)가 같았습니다. 실제로는 π0가 20%p 높습니다(30.4). 물체 질량을 실물에 가깝게 고쳐도(종이컵 70→10 g 등) 성공률이 유의하게 바뀐 정책은 없었습니다. Marker in Cup은 r 0.984로 그대로 맞았고, Cup in Bowl은 0.45로 올랐지만 여전히 맞지 않았습니다(21장). 세 번째 예제 영상 Fruits.mp4로 복원한 Serve Fruits는 순위가 실제와 같았지만(MMRV 0), GR00T가 52%로 높아 r은 0.824였습니다. 원인은 방해물인 주황 접시였습니다. 이 접시만 저자 것으로 바꾸자 GR00T가 4/25로 떨어졌고, r은 0.994로 논문(0.960)에 맞았습니다(22장). 정책 학습 개선(+17/21/40%)은 학습 코드와 실제 로봇이 필요해서 재현할 수 없습니다.

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

전제 조건: ComfyUI(127.0.0.1:8189)가 실행 중이어야 합니다. Ollama는 래퍼가 필요할 때 띄웁니다. VLM은 Pipeline A가 `qwen3.5:9b`(thinking 끔), B와 C가 `qwen3-vl:8b-instruct`입니다(16장). 두 모델을 모두 받아 둡니다(`ollama pull qwen3.5:9b`). 없으면 래퍼가 시작할 때 멈춥니다. 3~13장의 A 결과를 같은 VLM으로 재현하려면 `SIMFOUNDRY_OLLAMA_MODEL=qwen3-vl:8b-instruct`를 줍니다(`rerun_v2.sh`는 이미 고정). `--video-fpath`는 절대 경로로 줍니다(스테이지가 다른 작업 디렉터리에서 실행됨). Ollama 실행 파일이나 모델 저장소가 기본 위치가 아니면 `OLLAMA_BIN`, `OLLAMA_MODELS`로 지정합니다. 체크아웃에 `local/env.sh`가 있으면(머신별 브랜치, 예: `hanbin5/local`) 래퍼가 먼저 읽어서 이런 값을 채웁니다.

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
| `simfoundry/models/local_backends.py` (16장 추가분) | `SIMFOUNDRY_OLLAMA_THINK`: `0`이면 Ollama에 `think: false`, `1`이면 `true`. 설정하지 않으면 보내지 않음(예전과 같음) |
| hunyuan env `basicsr/data/degradations.py` | `functional_tensor` → `functional` |
| `simfoundry/utils/prompt_utils.py`, `11_make_objects_sim_ready.py`, `B_augmentation/stages/4_make_cousins_sim_ready.py` (20장) | 질량 프롬프트에 부피가 무엇인지 밝힘(`envelope_note`), 닫힌 얇은 껍질의 재료 부피(`mass_prompt_volumes`). 인자 없이는 원래 프롬프트와 같음 |
| `simfoundry/utils/prompt_utils.py` (`welded_volume`), `11_make_objects_sim_ready.py`, `B_augmentation/stages/4_make_cousins_sim_ready.py` (20.5) | 닫히지 않았다고 판정된 메시를 UV 이음매의 꼭짓점을 합쳐 다시 재고, 볼록 껍질의 40% 이상을 채우는 꽉 찬 물체면 그 부피를 질량 프롬프트에 줌(`weld_seams`). 인자 없이는 예전 부피 |
| `C_application/stages/2_teleop_og_scene.py` (20장) | 장면에 기록된 질량 적용(`apply_scene_masses`, 평가 단계와 같음) |
| `C_application/stages/1_eval_policy_og_scene.py`, `scripts/cfg/real2sim_cfg.yaml` (20장) | `s15_eval.tonemap_op`: 정책 카메라의 RTX 톤매퍼 선택. 기본 null(예전과 같음) |
| `A_reconstruction/stages/14_create_og_scene.py`, `scripts/cfg/*.yaml` (20장) | `s14_og.robot_standoff_m`(설정 기본 0.6 m): 로봇 베이스를 물체 중심 뒤에 둠. null이면 고정 위치 |
| `simfoundry/utils/og_utils.py` (`set_obj_materials`), `14_create_og_scene.py`, `B_augmentation/stages/6_sample_reconstructed_scene.py` (20.4) | 마찰 재질을 충돌 메시에 실제로 붙이고 운동 마찰도 같은 값으로 정함. 14단계와 B6은 stopped 블록 안에서 붙임 |
| `patches/BEHAVIOR-1K-link-physics-materials.patch` (신규), `scripts/installation/git_safe.sh`, `install_simfoundry.sh` (20.4 후속) | OmniGibson이 `link_physics_materials` 인자를 복사해서 쓰게 하는 패치와, 그 패치를 고정 커밋 위에 붙이는 설치 단계(`git_safe_checkout_patched`) |
| `simfoundry/utils/og_utils.py` (`set_obj_materials`, 20.4 후속) | 붙인 마찰을 init info의 `link_physics_materials`에도 기록. 패치된 OmniGibson일 때만 |

## 10. 추가 실험: 이미지 백엔드 비교 (2026-10-10)

Qwen-Image-2.1은 텍스트→이미지 전용이 아니라 생성·편집 통합 모델입니다. 칠한 표시·마스크로 국소 편집을 지정할 수 있고 RGBA를 직접 생성합니다. 그래서 파이프라인이 실제로 보내는 입력·프롬프트 6개로 세 모델을 비교했습니다 (`repro/q21/compare_image_backends.py`, 비교 이미지 `diagnostics/q21/sheet_all.jpg`).

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

구성: 이미지 모델 **Qwen-Image-2.1-Turbo** (`SIMFOUNDRY_IMAGE_BACKEND=qwen21`), 물체 제거는 **원본 기본 순서 `[mask, outline, bbox]`**(오버라이드 없음), **원본 `facebook/sam3`**. 12 GB에 맞추려고 남긴 조정은 `depth_pro` PDA 백엔드, Hunyuan `low_vram`, 순차 실행입니다. 응답 캐시는 별도 폴더를 썼습니다(캐시 키에 백엔드가 없어 v1 이미지가 재사용될 수 있음). 출력: `Data/put_marker_in_cup_v2`, `Data/put_cup_in_bowl_v2`. 실행: `repro/rerun_v2.sh` (두 장면에 A와 C, 이어서 PutMarkerInCup에 B. 단계별 로그는 `${SIMFOUNDRY_LOG_DIR:-~/simfoundry_logs}/v2_*.log`).

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

## 12. 조작 시연: 복원 장면에서 로봇이 실제로 물체를 옮김 (2026-10-11)

복원한 장면이 실제로 조작 가능한지 보여 주려고, 학습된 정책 없이 스크립트 동작으로 Franka가 물체를 집어 옮기게 했습니다(`repro/scripted_pick_place.py`).
- **성공 판정**: 사람 눈이 아니라 SimFoundry의 `PickPlaceTask` 목표 조건(`InsideAABB`, volume_threshold 0.5)으로 했습니다.
- **로봇 설정**: 장면 파일 그대로 썼습니다. IK 컨트롤러(`pose_delta_ori`)와 assisted grasping입니다.

| 장면 (v2) | 태스크 | 동작 | 결과 |
|---|---|---|---|
| PutMarkerInCup | B 단계가 제안한 `pick_marker_and_place_in_cup` | 마커 가운데를 위에서 잡기 → 들기 → 긴 축을 옆으로 돌리기 → 세우기(손바닥 옆) → 컵 위에서 놓기 | **성공**. 마커가 컵 안에 기대어 섬(최종 z 8.2 cm, 컵 높이 11.3 cm). 950스텝, 63 s |
| PutCupInBowl | 직접 작성(B 형식, `InsideAABB(cup, bowl)`, `repro/tasks/put_cup_in_bowl_v2_place_cup_in_bowl.yaml`). 이 장면은 B를 돌리지 않음 | 컵 가장자리 벽 집기(윗지름 9.4 cm > 그리퍼 폭 8 cm) → 들기 → 그릇 위 → 안으로 내려놓기 | **성공**. 컵이 그릇 안에 똑바로 섬. 704스텝, 46 s |

영상: `results_view/demo_marker_in_cup.mp4`, `results_view/demo_cup_in_bowl.mp4`.

다시 실행 (simfoundry 환경, 저장소 루트에서). 경로 인자(`--task`, `--out`, `--background`, `--scene-file`)는 절대 경로로 줍니다. 스크립트가 `scripts/cfg`로 옮겨 간 뒤 읽기 때문입니다.

```bash
python repro/scripted_pick_place.py --scene put_marker_in_cup_v2 \
  --task "$PWD/Data/put_marker_in_cup_v2/proposed_tasks/put_marker_in_cup_v2_pick_marker_and_place_in_cup.yaml" \
  --pick iter_2 --place iter_1 --mode insert_vertical --out "$PWD/results/results_view/demo_marker_in_cup.mp4"
python repro/scripted_pick_place.py --scene put_cup_in_bowl_v2 \
  --task "$PWD/repro/tasks/put_cup_in_bowl_v2_place_cup_in_bowl.yaml" \
  --pick iter_2 --place iter_0 --mode rim_pinch --out "$PWD/results/results_view/demo_cup_in_bowl.mp4"
```

2026-10-16에 이 두 명령을 다시 돌려 기록과 같은 결과를 확인했습니다. 컵은 704스텝 성공으로 물체 최종 위치까지 같았고, 마커는 950스텝 성공에 최종 위치가 6 mm 달랐습니다(물리 시뮬레이션이 실행마다 조금씩 다른 것으로 보임).

아래 변형은 같은 명령에 옵션을 더한 것입니다. 배경 `--background "$PWD/assets/backgrounds/mesh_backgrounds/droid_v1.background.json"`, 원본 시점 `--real-view`(렌즈를 넓히려면 `--real-view-hfov 90`), 7단계 장면 파일 `--scene-file`(13.3).

**배경 추가**: 저자들이 공개한 DROID 작업대 배경 메시(`assets/backgrounds/mesh_backgrounds/droid_v1`)를 붙여 다시 찍었습니다.
- **배치 방법**: 배경 메타데이터(`droid_v1.background.json`)에 등록된 기준을 그대로 썼습니다. 로봇과 방의 상대 위치는 `default_robot` 기준, 책상 면 높이는 `support_surface.canonical_height_m = 0` 기준입니다. 책상 면이 우리 물체가 놓인 z = 0과 일치하므로 로봇 베이스는 기준대로 z = -0.04에 둡니다.
- **카메라**: 실제 DROID 배치(로봇 좌우 2대, `nv_franka_droid_high_res`)를 썼습니다.
- **결과**: 두 태스크 모두 다시 **성공**했습니다(마커 942스텝, 컵 681스텝). 영상은 `results_view/demo_marker_in_cup_bg.mp4`, `results_view/demo_cup_in_bowl_bg.mp4`입니다.
- **주의**: 이 배경은 동영상에서 복원한 것이 아니라 저자들이 미리 스캔해 공개한 것입니다. 예제 동영상의 책상·화이트보드·로봇 거치대가 같은 작업대로 보여 함께 썼습니다. 동영상에서 배경을 자동 복원하는 기능(VOID 인페인팅 + Gaussian splat)은 나중에 따로 실행했습니다(13장).

**확인된 것**
- 복원 물체의 충돌 형상이 조작에 쓸 만합니다. 컵·그릇은 안이 빈 용기로 만들어져 마커와 컵이 그 안에 들어갔습니다.
- 크기·질량·마찰로 잡기와 운반이 됐습니다.
- B 단계가 제안한 태스크의 판정 조건이 그대로 동작했습니다.

**조정한 점**
- **로봇 위치**: 이 동영상들은 로봇 없이 촬영돼, 파이프라인이 로봇을 물체에서 0.87 m(PutMarkerInCup) 떨어진 곳에 임의로 두었습니다. Franka 팔이 닿지 않는 거리입니다. 저장소가 B와 C 사이에 두는 수동 장면 편집처럼, 장면 JSON의 `robot0.root_link.pos`를 물체 앞 0.55 m로 옮겼습니다.

**한계**
- 사람이 설계한 웨이포인트 동작이라 정책 성능과는 무관합니다.
- 장면당 1회 시연이므로 성공률이 아닙니다.
- 논문 Table G.1(공개 π0/π0.5 정책의 시뮬 성공률)은 18장에서 저장소 평가 코드(`C_application/stages/1_eval_policy_og_scene.py`, `simfoundry/policies/openpi.py`)로 재현했습니다.

**디버깅 메모** (진단 스크립트: `repro/diag/diag_*.py`)
- 리셋 자세의 그리퍼는 수직에서 25° 기울어 있습니다. 접근 축은 반드시 hand→eef 방향(eef 로컬 +z)으로 잡아야 합니다.
- 대칭 그리퍼의 닫힘 축 부호(±)는 손목 회전이 작은 쪽을 골라야 합니다.
- 물체를 세울 때 손바닥이 앞을 향하면 손목이 베이스 쪽으로 접혀 관절 한계에 걸립니다. 옆을 향하게 해야 합니다.

**원본 영상과 같은 시점의 비교** (`repro/compose_real_vs_sim.py`, `repro/scripted_pick_place.py --real-view`)
- **카메라 배치**: SimFoundry는 3단계에서 고른 입력 프레임의 카메라를 월드 기준으로 삼습니다(4단계 `s4_frame/image_<i>_cam2world.npy`). 그래서 이 카메라를 그대로 시뮬레이터에 두었습니다. OpenCV 축을 USD 카메라 축으로 바꿨고, 화각은 DA3 내부 파라미터(PutMarkerInCup 63.6°, PutCupInBowl 60.0°)로 맞췄습니다.
- **정렬**: 같은 시점의 렌더와 실제 프레임을 겹치면 물체 위치·크기가 거의 그대로 포개집니다(`diagnostics/realview_alignment.jpg`). 복원 결과가 실제 장면과 기하적으로 일치한다는 직접적인 확인입니다.
- **영상 구성** (3분할, 1920×360): `results_view/real_vs_sim_marker_in_cup.mp4`(79 s), `results_view/real_vs_sim_cup_in_bowl.mp4`(63 s).
  - 왼쪽: 원본 입력 영상 전체, 이후 기준 프레임(117 / 241)에서 고정.
  - 가운데: 복원 물체 하나씩 360° 턴테이블(`repro/turntable.py`). 시뮬레이터와 같은 Isaac 렌더러, 고도 30°, 물체당 96프레임이고 물체 이름과 복원 크기를 표시합니다.
  - 오른쪽: 같은 시점의 복원 장면, 이후 로봇 시연.
  - 이 시점에서도 두 태스크 모두 판정 성공했습니다.
  - 턴테이블에서 보이는 점: 텍스처가 뒷면까지 이어집니다(컵의 잎 무늬와 QR, 마커 뚜껑). 다만 글자는 뭉개진 가짜 글자이고, 컵·그릇 안쪽은 비어 있어 용기로 쓸 수 있습니다.
- **배경**: 이 영상에는 넣지 않았습니다. 공개 배경은 저자들이 로봇까지 함께 스캔한 DROID 작업대 메시(Station 1·4, YAM 작업대)뿐이고, Gaussian splat 배경은 공개 데이터셋에 없습니다. 입력 동영상으로 로봇 없는 splat 배경을 만드는 저장소 경로(auto_bg)는 13장에서 실행했습니다.

## 13. 자동 3DGS 배경: 논문 파이프라인으로 로봇 없는 배경 만들기 (2026-10-12)

논문 App. E.5.1의 자동 배경 기능입니다. 저장소 경로는 `scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/`의 7단계(`run_auto_align.sh`)입니다. 입력 동영상에서 책상 위 물체를 지운 영상을 만들고, 그 영상으로 Gaussian splat 배경을 학습해 OmniGibson 장면에 넣습니다. 같은 단계 스크립트를 같은 순서로 `repro/run_auto_bg.sh`가 호출합니다.

- **장면**: PutMarkerInCup. auto_bg 입력을 만드는 splat-prep 모드로 1–14단계를 다시 돌린 `put_marker_in_cup_bg`(160프레임, 672×384)입니다.
- **프레임 수**: 저장소 기본값은 400프레임입니다. VOID가 이 GPU에서 프레임당 약 25초 걸려, 400프레임이면 VOID만 약 3시간이 걸립니다.

### 13.1 단계별 결과

| 단계 | 하는 일 | 시간 | 결과 |
|---|---|---|---|
| 1 quadmask | 지울 물체 목록 → SAM3 키프레임 마스크 → SAM2 영상 전파 | 1.9분 | 마커·컵·가위. 프레임당 마스크 비율 3–11% |
| 2 VOID pass1 | CogVideoX-Fun 5B 영상 인페인팅(void_pass1), 160프레임 한 덩어리 | 26.5분 | 앞 절반은 깨끗함. 근접 촬영 후반에 가위 테·마커 뚜껑 잔상 |
| 3 VOID pass2 | pass1에서 만든 warped noise로 85프레임 창 2개(0–84, 75–159)를 다시 생성 | 43분 | 잔상 대부분 제거. 100–159프레임에 가위 날 일부 남음 |
| void DA3 | 지운 프레임의 깊이·카메라 | 1.3분 | 앵커 잔차 평균 2.6 mm, 회전 0.59° |
| 4 seed | void DA3를 원본 DA3에 강체 정렬 후 점 20만 개 | 4초 | 카메라 중심 잔차 4.6 mm(최대 12 mm), 축척 차 1% |
| 5 splat | splatfacto-big, 깊이 손실 0.5, SO3xR3 카메라 보정, **80k 반복(저장소 기본값)** | 43분 | 가우시안 388k. 지운 프레임 대비 PSNR 23.2 dB(20프레임 평균). 렌더 깊이는 원본 DA3보다 1% 깊음 |
| 6 bridge | DA3 → OG 월드 강체 변환(사이드카) | 3초 | 축척 1.000 |
| 7 assets | 3dgrut로 NuRec USDZ 변환, 장면 JSON에 `gs_background` 추가 | 9초 | `assets/scenes/put_marker_in_cup_bg/put_marker_in_cup_bg_scene_state_auto_bg.json` |

30k 반복으로 먼저 학습한 결과(PSNR 23.3 dB, 가우시안 468k)와 80k 결과는 학습 시점에서 거의 같았습니다. 최종 결과는 80k입니다.

### 13.2 12 GB와 로컬 VLM에 맞춰 바꾼 점

- **void 환경**: 저장소 설치 스크립트(`install_void.sh`)는 numpy를 1.26.4로 고정하지만 pyarrow는 고정하지 않습니다. 그래서 numpy 2 전용인 pyarrow 26이 깔려 transformers → sklearn import에서 실패했습니다. pyarrow 20.0.0으로 고정했습니다(이제 `install_void.sh`가 고정함).
- **1단계 물체 목록**: Gemini 대신 쓴 Qwen3-VL 8B가 키프레임 0에서 컵 옆 마커를 놓쳤습니다. 그대로 두면 "빈 책상" 배경에 마커가 남습니다.
  - 1단계에 선택 설정 `s1_quadmask.objects`를 추가했습니다(기본 null이면 원래 동작).
  - 드라이버가 이 값에 정규 재구성 5단계의 물체 목록을 넣습니다. 같은 프롬프트와 같은 VLM으로 분해용 프레임에서 찾은 목록입니다.
  - 로컬 VLM은 seed가 고정돼 재시도해도 같은 답을 냅니다.
- **VOID를 12 GB에서 돌리기** (`VOID_LOW_VRAM=1`일 때만, `patches/void-model-low-vram.patch`. `install_void.sh`가 `void-model.patch` 다음에 적용):
  - pass1은 확산 단계는 들어가지만 마지막 VAE 디코딩에서 메모리가 부족했습니다(11.2 GB 사용 중). VAE를 타일로 나눠 디코딩합니다.
  - pass2 스크립트는 5B transformer와 T5를 GPU에 통째로 올린 뒤 모델 단위로 오프로딩합니다. 이를 pass1처럼 층 단위 CPU 오프로딩과 타일 디코딩으로 바꿨습니다.
- **pass1 한 덩어리 꼬리 문제**: 160프레임은 한 덩어리로 처리됩니다. 그런데 CogVideoX는 4k+1 길이만 남겨 157프레임을 돌려주고, pass2의 두 번째 창(75–159)이 범위를 벗어나 실패했습니다.
  - `2_run_void_pass1.py`가 마지막 프레임을 반복해 프레임 수를 맞춥니다. 그 프레임들은 pass2가 입력 영상에서 다시 생성합니다.
  - 저장소 기본값(400프레임, 덩어리 173 = 4k+1)에서는 생기지 않는 경우입니다.
- **pass2 재개**: `s3_pass2.reuse_chunks`를 추가했습니다(기본 false). 끝난 창을 다시 계산하지 않고 이어서 돌립니다.
- **void DA3**: 저장소의 2b(220프레임 덩어리 연쇄, 24 GB)는 12 GB에서 16프레임 덩어리가 되어 20° 이상 틀어졌습니다. 그래서 정규 2단계와 같은 `repro/da3_anchored_stage.py`(기준 호출 + 앵커 정렬)를 썼습니다.

### 13.3 시뮬레이션에 넣기

- **불러오기**: `repro/scripted_pick_place.py --scene-file <..._auto_bg.json>`로 7단계 장면을 그대로 씁니다.
  - splat은 보이기만 하는(visual-only) 물체라서 바닥 평면은 충돌용으로만 남기고 숨깁니다.
  - 7단계가 물체를 `USDObject`로 다시 등록해 `native_bbox`가 없습니다. 그래서 물체 기준 바운딩 박스로 대신했습니다.
- **책상 높이**: 렌더링한 splat 깊이는 원본 DA3보다 약 1% 깊습니다. 앵커 시점에서 물체가 놓인 z = 0보다 1 cm 안쪽으로 아래라서 바닥 평면을 따로 고치지 않았습니다.
  - 가우시안 중심만 보면 −2.5 cm로 나오지만, 평평하고 큰 가우시안이 표면 아래에 걸쳐 있어서 생기는 착시입니다.
- **톤매퍼**: Isaac Sim 기본 톤매퍼 ACES(op 6)가 NuRec splat을 복사광처럼 처리해 밝기와 채도를 올렸습니다. 실제 앵커 프레임과 배경 화소를 비교했습니다.

  | 톤매퍼 | 평균 절대 오차 | 평균 색 (실제 159/130/108) |
  |---|---|---|
  | ACES (기본) | 0.126 | 183/158/131 |
  | **linear (op 1)** | **0.032** | 154/124/103 |
  | clamp (op 0) | 0.170 | 134/98/75 |

  - 배경이 있으면 시연에서 linear로 바꿉니다. 조명을 받는 복원 물체도 자연스럽게 보입니다.
  - OmniGibson과 SimFoundry는 톤매퍼를 설정하지 않습니다. 7단계가 참조하는 nv_desk 장면의 조명 설정(`lighting_state`)은 공개 자산에 없습니다.
- **결과**: 같은 B 단계 태스크(`InsideAABB(marker, cup)`)가 이 배경 장면에서도 **성공**했습니다(864스텝, 30k·80k 배경 모두). 원본 카메라 시점에서 책상 모서리·벽·거치대·물체 위치가 실제 프레임과 겹칩니다.
- **영상**: `results_view/real_vs_sim_gsbg_marker_in_cup.mp4`(74초, 3분할)와 정지 화면 `.jpg`입니다.
  - 왼쪽: 원본 영상 → 앵커 프레임 127.
  - 가운데: 이 재구성의 물체 360° 턴테이블.
  - 오른쪽: 자동 3DGS 배경 위 시뮬레이션. 90° 렌즈이고, 노란 상자가 실제 카메라 시야입니다.

### 13.4 한계

- **가위 날 잔상**: VOID가 100–159프레임에 남긴 가위 날 일부가 splat에도 그 자리에 들어갔습니다. 시뮬레이션에서는 같은 자리에 놓인 복원 가위가 대부분 가립니다.
- **촬영되지 않은 영역**: 90° 렌즈는 휴대폰이 찍지 않은 영역(왼쪽 가장자리, 위쪽)까지 보여서 그곳에는 뿌연 가우시안(floater)이 보입니다. 노란 상자 안(실제 카메라 시야)은 깨끗합니다.
- **책상 밖 물체**: 배경에는 책상 위 물체만 빠지고 방의 나머지는 남습니다. 왼쪽의 로봇 거치대도 남는데, 방의 일부라서 그렇습니다.
- **로봇 카메라 시점 (2026-10-16 확인)**: 로봇의 외부 카메라로 렌더하면 이 배경은 줄무늬와 부유물로 무너져 쓸 수 없습니다. 위 결과는 촬영 카메라 시점에서만 확인한 것입니다(17.4). 재학습으로 나아진 정도는 17.7에 있습니다.
- **두 번째 장면 미수행**: PutMarkerInCup만 했습니다(17장에서 두 장면 모두 실행). PutCupInBowl은 splat-prep 정규 재구성(약 1시간)과 VOID(약 1.5시간)가 더 필요합니다.
- **디스크**: void 환경 8.1 GB, nerfstudio 환경 15 GB, 3dgrut 환경 14 GB, VOID 가중치 41 GB(`deps/void-model`, 게이트 없음), 장면 데이터 2.9 GB입니다.

### 13.5 수정·추가한 파일

| 파일 | 변경 |
|---|---|
| `auto_bg_reconstruction/1_generate_quadmask_for_void.py`, `scripts/cfg/auto_bg.yaml` | 선택 설정 `s1_quadmask.objects` (주어지면 VLM 물체 제안 대신 사용. 이때는 gcloud 프로젝트도 조회하지 않음) |
| `auto_bg_reconstruction/2_run_void_pass1.py` | 한 덩어리 출력이 짧으면 마지막 프레임으로 채움 |
| `auto_bg_reconstruction/3_run_void_pass2.py`, `scripts/cfg/auto_bg.yaml` | 선택 설정 `s3_pass2.reuse_chunks` |
| `patches/void-model-low-vram.patch` (신규), `scripts/installation/install_void.sh` | `VOID_LOW_VRAM=1`: VAE 타일 디코딩, pass2 층 단위 CPU 오프로딩. 설치 스크립트가 적용 |
| `scripts/installation/install_void.sh` | pyarrow 20.0.0 고정 (26.0.0이 깔리면 numpy 1.26.4에서 import 실패) |
| `repro/run_auto_bg.sh` (신규) | 7단계 드라이버 (환경 변수, 1단계 물체 목록, 앵커 DA3) |
| `repro/da3_anchored_stage.py` (신규) | void DA3: 덩어리 연쇄(2b) 대신 기준 호출 + 앵커 정렬 |
| `repro/scripted_pick_place.py` | `--scene-file`, splat 배경일 때 바닥 숨김과 linear 톤매퍼, `USDObject` 바운딩 박스 |
| `repro/compose_real_vs_sim.py` | 프레임 폴더를 DA3 프레임 수로 선택, 시뮬레이션 패널 메모 |
| `repro/tasks/put_marker_in_cup_bg_pick_marker_and_place_in_cup.yaml` | v2에서 B 단계가 만든 태스크를 같은 물체 이름으로 재사용 |

다시 실행:

```bash
repro/run_auto_bg.sh put_marker_in_cup_bg "desk, table, or counter"
```

## 14. 결과와 체크포인트 (Google Drive)

git에는 코드와 이 기록만 넣습니다. 영상·이미지·로그, 장면 데이터, 체크포인트는 Google Drive의 [`Inner-King/innerinner/`](https://drive.google.com/drive/folders/1lmzFWqMATc52zW7gyiog0YZM3ewKDUzY) 폴더에 둡니다. 링크가 있으면 누구나 볼 수 있고, 폴더의 `README.md`에 장면별 내용이 있습니다.

| Drive 경로 | 내용 | 체크아웃 쪽 |
|---|---|---|
| `results/results_view/` | 최종 영상·이미지 (12·13장) | `results/results_view/` |
| `results/results_view/policy_eval/` | Table G.1 정책 평가 요약: 실행별 표(`runs.md`), 과제·장면별 r·MMRV(`agreement.md`), 실행마다 성공·실패 영상 하나씩. 아래 두 원본 폴더를 읽어 `repro/policy_eval_ours/results_view.py`가 만듦 | `results/results_view/policy_eval/` |
| `results/policy_eval_ours/`, `results/logs/policy_eval/` | 정책 평가 원본(평가 단계의 깊은 폴더 그대로): 앞은 19–23장, 뒤는 18장의 π0·π0.5 | `results/policy_eval_ours/`, `~/simfoundry_logs/policy_eval/` |
| `results/diagnostics/` | 점검 이미지(`realview_alignment.jpg` 등), 이미지 백엔드 비교(`q21/`) | `results/diagnostics/` |
| `results/logs/` | 설치·실행 로그 (A/B/C, 시연, auto_bg) | `${SIMFOUNDRY_LOG_DIR:-~/simfoundry_logs}` |
| `results/model_calls/` | VLM·이미지 모델 응답 캐시. `--cache-mode`로 같은 응답을 다시 씀 | `.cache/simfoundry/` |
| `results/scenes/<scene>.tar` | 장면별 파이프라인 산출물. 저장소 루트에서 풀면 `Data/<scene>/`가 됨(13장 장면은 `assets/scenes/<scene>/`도) | `Data/<scene>/` |
| `checkpoints/` | 우리가 학습한 가중치. 지금은 13장의 배경 splat (splatfacto 30k·80k 반복). 학습 run 폴더를 통째로 `put_marker_in_cup_bg/splatfacto_{80k,30k}/<타임스탬프>/`에 둠(`config.yml`, `nerfstudio_models/*.ckpt` 등) | `Data/put_marker_in_cup_bg/auto_bg/{splat,splat_30k}/outputs/ns_data/splatfacto/<타임스탬프>/` |

- 장면 tar에는 `*.ckpt`를 넣지 않습니다. 시뮬레이션에 쓰는 splat 결과(`auto_bg/splat/export/*.ply`, 7단계 USDZ)는 tar에 있습니다. 학습을 이어 가려면 장면을 받은 뒤 `pull checkpoint`로 run 폴더를 제자리에 받습니다(아래 예).
- 배경 관련 장면(`put_marker_in_cup_bg`, `put_cup_in_bowl_v2_bg`, `put_marker_in_cup_v2_bg`)에는 단계가 입력을 절대 경로로 걸어 둔 심볼릭 링크가 그대로 들어 있습니다(예: `auto_bg/splat_30k/ns_data/images/*.png -> /home/inner/real2sim/SimFoundry/Data/<scene>/…`). 링크 대상은 같은 tar에 들어 있으므로, `pull scene`이 압축을 푼 뒤 이 링크를 풀어 놓은 사본을 가리키는 상대 링크로 바꿉니다. 장면 밖을 가리키는 링크는 바꾸지 않고 목록만 보여 줍니다.
- 밖에서 받는 사전 학습 가중치(Hunyuan3D-2.1, DA3, SAM 2.1/3, VOID, Ollama·ComfyUI 모델)는 올리지 않습니다. 원래 배포처에서 받습니다(일부는 게이트가 있음).
- 올리고 받기는 `repro/drive_sync.sh`(rclone)로 합니다. 올릴 때는 복사만 하고 Drive에서 지우지 않습니다.

```bash
repro/drive_sync.sh push results                 # results/ -> Drive results/
repro/drive_sync.sh push logs
repro/drive_sync.sh push scene put_marker_in_cup_bg
repro/drive_sync.sh push checkpoint Data/put_marker_in_cup_bg/auto_bg/splat/outputs/ns_data/splatfacto/2026-10-12_011411 put_marker_in_cup_bg/splatfacto_80k/2026-10-12_011411
repro/drive_sync.sh pull scene put_marker_in_cup_v2  # 받아서 저장소 루트에 풂
repro/drive_sync.sh pull checkpoint put_marker_in_cup_bg/splatfacto_80k/2026-10-12_011411 Data/put_marker_in_cup_bg/auto_bg/splat/outputs/ns_data/splatfacto/2026-10-12_011411
repro/drive_sync.sh ls results
```

rclone 원격은 한 번 만들어 둡니다. 이름은 `innerking`이고, 루트는 Inner-King 폴더입니다(폴더 ID는 Drive 주소의 `folders/` 뒤).

```bash
rclone config create innerking drive scope=drive root_folder_id=<Inner-King 폴더 ID>
```

## 15. `repro/` 파일과 그 밖의 실험 기록

| 파일 | 용도 | 장 |
|---|---|---|
| `run_local_pipeline.sh`, `start_ollama.sh` | 로컬 백엔드로 A/B/C 실행 | 8 |
| `eval_pose_fit.py` | 포즈 정합 잔차 | 3.2, 6, 11, 17 |
| `compare_scenes.py` | 저자 공개 장면과 크기·거리 비교 | 3.2 |
| `compare_s5_runs.py` | 두 5단계 결과를 반복마다 비교: 기록, 마스크·이미지의 픽셀 일치, 깊이 차이 | 24 |
| `q21/compare_image_backends.py` | 이미지 백엔드 비교 | 10 |
| `rerun_v2.sh` | v2 재실행 | 11 |
| `scripted_pick_place.py`, `tasks/` | 스크립트 조작 시연과 그 태스크. `--wrist`는 손 카메라도 녹화 | 12, 13, 17, 30 |
| `turntable.py`, `compose_real_vs_sim.py` | 물체 턴테이블 렌더, 3분할 비교 영상 | 12, 13 |
| `run_auto_bg.sh`, `da3_anchored_stage.py` | 자동 3DGS 배경 (`VOID_DA3=canonical`, `AUTO_BG_STOP_BEFORE_ASSETS=1`은 17장) | 13, 17 |
| `drive_sync.sh` | Drive로 올리고 받기 | 14 |
| `vlm_compare/` | 로컬 VLM 비교: 파이프라인의 VLM 호출 재생·채점, 좌표 순서 시험, 11단계 질량 프롬프트 변형(`MASS_PROMPT`)과 부피 후보(`mass_volumes.py`) | 16, 20.1, 20.5 |
| `policy_eval/` | Table G.1 시뮬 평가: Colab 런타임 준비, openpi·DreamZero 서버와 평가 드라이버, DreamZero 클라이언트 점검, 평가용 태스크 설정과 시작 상태, 시작 상태 생성기, 요약·롤아웃 분석, 비교 실험(`ablations/`) | 18, 19.4 |
| `policy_eval_ours/` | 우리 복원 장면을 저자 DROID 책상에 올리기(물체 질량·과제별 틀 장면 지정 포함), 장면 점검, GR00T N1.6 로컬 실행, π0·π0.5 로컬 실행(`run_local_openpi.sh`, 체크포인트 받기 `fetch_openpi_ckpt.py`), 과제별 r·MMRV와 그 95% 구간, 배경 스캔의 로봇 가리기(`mask_scanned_robot.py`)와 배경만 바꾼 장면 사본(`use_background.py`), 평가 결과 요약 폴더(`results_view.py`) | 19, 21, 22, 23, 30 |
| `friction/check_friction.py` | 물체 마찰이 PhysX에 들어가는지 점검: 붙이기 전후의 shape 재질, 미끄럼 시험(바닥 결합 방식 포함), `link_physics_materials` 저장, PhysX 질량 | 20.4, 29.5 |
| `mass_friction/` | 11단계 질량 벤치마크: YCB 받기·렌더(`ycb_fetch.sh`, `render_ycb.py`), 방법별 채점(`bench_mass.py`) | 29 |
| `friction/check_saved_scene.py`, `friction/check_finger_friction.py` | 마찰을 기록한 장면을 텔레옵처럼 로드해 PhysX 재질과 다시 저장을 점검, 로봇 손가락 마찰 점검 | 20.4 |
| `diag/diag_*.py` | 조작 시연을 만들며 쓴 로봇 베이스·IK·하강·그리퍼 축 진단 | 12 |
| `diag/smoke_hunyuan_12gb.py` | Hunyuan3D-2.1 형상·텍스처가 12 GB에 들어가는지 점검 | 2 |
| `g4/setup_g4.sh`, `g4/constraints_simfoundry.txt` | Colab G4 런타임 설치(Vulkan·libGLU 보완, 버전 고정) | 17 |
| `g4/rerun_g4.sh`, `g4/auto_bg_g4.sh` | 12 GB 조정 없이 A/B/C, 저장소 기본값으로 auto_bg | 17 |
| `g4/compare_runs.py` | 두 실행의 단계별 시간, 물체 크기·질량, 정합 잔차 비교 | 17 |
| `g4/compare_da3_tracks.py`, `g4/splat_depth_scale.py` | void-DA3 궤적과 원본 DA3 비교, 배경 splat의 깊이 축척 | 17 |
| `g4/eval_splat.sh`, `g4/ns_run.sh` | 배경 splat의 학습 제외 프레임 PSNR·SSIM·LPIPS (`ns-eval`) | 17 |
| `g4/render_bg_views.py`, `g4/check_bg_variant.sh` | 배경 장면을 실제 카메라와 로봇 카메라 4대로 렌더 | 17.7 |
| `g4/retrain_bg.sh`, `g4/scale_void_da3.py`, `g4/crop_bg_splat.py`, `g4/carve_bg_splat.py` | 배경 splat 재학습(A100)과 후처리 | 17.7 |
| `g4/wrist_sheet.py` | 배경별 시연 영상의 외부·손목 카메라 화면을 나란히 놓은 그림 | 30.2 |
| `masked_bg/` | 중단한 배경 시도 (아래) | |
| `fruit_slicing/` | 진행 중인 과일 자르기 장면 (아래) | |

**중단한 시도: 물체를 마스크로 가리고 배경 splat 학습 (2026-10-11)**. 13장 전에, VOID로 물체를 지우는 대신 물체 픽셀을 SAM3 마스크로 가려 splat의 광도·깊이 손실과 seed 점구름에서 빼는 방식을 먼저 준비했습니다.
- 스크립트: `masked_bg/make_object_masks.py`(프레임별 물체 마스크), `masked_bg/dense_da3_anchored.py`(조밀 DA3를 OmniGibson 월드에 앵커 정렬), `masked_bg/align_da3_to_og.py`(조밀 DA3 → OG 월드 Sim(3)과 책상 높이 점검).
- 산출물: `Data/put_marker_in_cup_v2_bg`, `Data/put_cup_in_bowl_v2_bg`. 프레임, 마스크, DA3(PutMarkerInCup은 Sim(3) 정렬까지)만 있고 splat 학습 결과는 남아 있지 않습니다. 그 뒤 저장소의 auto_bg 경로(13장)로 진행했습니다.

**진행 중: 과일 자르기 장면 (2026-10-15)**. `fruit_slicing/make_slicing_scene.py`는 저자 공개 장면 `droid_desk_serve_fruits`로 과일 자르기 장면을 만드는 스크립트입니다(3dgrut 환경).
- 과일 4개를 평면으로 잘라 반쪽 메시를 만듭니다. 단면 캡과 충돌용 볼록 껍질이 포함됩니다.
- 칼과 칼꽂이를 넣고, 장면 JSON과 자르기 계획(`cut_plan.json`)을 씁니다.
- 아직 미완성입니다. 이 장면을 쓰는 시연 스크립트(`slice_fruits_demo.py`)는 없고, 출력 장면(`assets/scenes/DROID/droid_desk_fruit_slicing`)을 만든 흔적도 없습니다.

## 16. 로컬 VLM 비교: Pipeline A는 Qwen3.5-9B로 (2026-10-16)

Qwen3.5-9B(Ollama `qwen3.5:9b`, Q4_K_M, 본체 5.6 GB와 비전 0.9 GB)는 공개 벤치마크에서 Qwen3-VL-8B보다 높게 나옵니다. 다만 그 수치는 thinking 모드로 보입니다. 파이프라인은 4096 토큰 한도 안에서 바로 답해야 하므로, 파이프라인의 실제 VLM 호출로 두 모델을 직접 비교했습니다.

- **방법** (`repro/vlm_compare/`)
  - 앞선 실행이 저장한 입력을 그대로 씁니다: 키프레임, 3단계 후보 시트, 5단계 반복 이미지, 메시와 정면 렌더, 장면 JSON.
  - 같은 프롬프트와 인자를 `Gemini(...)` → shim → Ollama 경로로 다시 보냅니다. `refine_with_vlm`, B7의 프롬프트 빌더와 파서 같은 스테이지 함수도 그대로 씁니다.
  - ComfyUI가 필요 없고, `Data/` 아래에는 아무것도 쓰지 않습니다.
  - 응답 캐시는 끕니다. 캐시 키에 Ollama 모델이 들어가지 않아서, 켜 두면 다른 모델의 답이 재생됩니다.
  - 박스 정답은 원래 실행의 SAM3 마스크입니다.
  - 같은 입력이면 qwen3-vl은 원래 실행과 같은 답을 냅니다. 3단계 선택, 8단계 정면 선택, 11단계 질량(3장의 가위 50 g·마커 36 g·컵 70 g)이 그대로 다시 나옵니다.
  - Pipeline A의 VLM 호출은 3·5·8·11단계뿐입니다(6단계 검사는 기본으로 꺼져 있음). 모두 재생했습니다.
- **thinking은 꺼야 합니다.** Qwen3.5는 기본이 thinking입니다.
  - 켜 두면 생각만으로 4096 토큰을 쓰고 잘립니다. 8호출 중 6호출이 한도에 걸렸고 4호출은 재시도해도 실패했습니다. 1·5단계에서는 단계가 예외로 멈추고, 한도에 걸린 호출은 120–167 s가 걸립니다.
  - 박스도 프롬프트가 요청한 [ymin,xmin,ymax,xmax] 순서로 씁니다. 그러면 shim의 순서 변환이 박스를 뒤집습니다(합성 도형 2/2).
  - 그래서 shim에 `SIMFOUNDRY_OLLAMA_THINK`를 추가했습니다(`0`이면 `think: false`).
  - thinking을 끄면 좌표 규약이 qwen3-vl과 같습니다. 박스 [x1,y1,x2,y2]와 점 [x,y](0–1000)를 쓰고 [y,x] 요청을 무시합니다(합성 도형 2장과 실제 프레임, 18/18).
  - qwen3-vl에 `think: false`를 보내도 답이 바이트 단위로 같습니다.

**Pipeline A 호출** (qwen3.5는 thinking 끔)

| 호출 | `qwen3-vl:8b-instruct` | `qwen3.5:9b` |
|---|---|---|
| 1단계(auto_bg) 키프레임 0 물체 목록 | 컵·가위. **마커 누락** | 3/3 |
| 같은 호출을 같은 영상 8프레임에 | 마커 4/8 (먼 시점 4장 누락) | 마커 6/8 |
| 3단계 프레임 선택, 후보 시트 3개 | v1 시트에서 4096 토큰 반복 루프 → 재시도로 답(120 s). 최선 1개, 차선 1개, 최하위 1개(마커가 가장 크다는 틀린 근거) | 루프 없음(약 20 s). 최선 2개, 차선 1개 |
| 5단계 검출 (3장면, 물체 18개, 마지막 빈 책상 9회) | 18/18, 오검출 0, 빈 결과 9/9 | 같음 |
| 5단계 박스 IoU (SAM3 대비 평균, 중심이 정답 박스 안) | 0.71, 18/18 | 0.76, 18/18 |
| 8단계 정면 선택 (물체 9개) | 9개 중 4개가 qwen3.5와 다름 | 다른 4개 중 3개(마커 2개, 가위)는 사진과 같은 방향의 뷰를 고름: 마커는 뚜껑 방향과 라벨 면, 가위는 날 끝. 렌더를 눈으로 보면 이쪽이 맞음. 무늬 있는 종이컵은 NONE 대신 A. 호출당 17 s → 6 s |
| 11단계 질량 | 종이컵 70·150·70 g | 종이컵 30·25·20 g(실물 약 10 g). **그릇 5.51 kg**(qwen3-vl 0.5 kg) |
| 22호출 합계(1·3·5단계) | 300 s, MAX_TOKENS 1회 | 192 s, 0회 |
| VRAM (호출이 더한 장치 메모리) | 7.8 GB, 이미지 4장이면 8.2 GB | 6.7 GB |

- 11단계는 qwen3.5가 프롬프트의 "chain of thought"를 실제로 써서 호출당 13–33 s가 걸립니다(qwen3-vl은 JSON만 내고 5 s).
- 그릇 5.51 kg은 스테이지가 넘긴 부피(닫히지 않은 메시라 바운딩 박스의 65%, 5.8 L)에 플라스틱 밀도를 곱한 값입니다. 속이 빈 그릇을 꽉 찬 물체로 계산했습니다. 컵은 속이 비었다고 보고 맞게 줄였습니다.
- 마커처럼 꽉 찬 물체는 두 모델 모두 부피 × 물 밀도로 답합니다(약 36–40 g, 실물 약 15–20 g).

**Pipeline B 호출**

| 호출 | `qwen3-vl:8b-instruct` | `qwen3.5:9b` |
|---|---|---|
| B1 부품 목록 (가위, 컵) | [Handle, Blade, Pivot Point], [Main Body, Rim, Logo, Text, Green Stripe, Bottom Ring] | [Handle], **[杯身]** (중국어) |
| B7 태스크 제안 (2장면 × seed 0·1000·2000, 온도 0.7) | 6/6 실행에서 5개씩 파싱. PutCupInBowl에서 "컵을 그릇에"가 2/3 실행에 나옴 | 4/6만 5개. PutCupInBowl은 2개, 0개 파싱(`task_name` 키가 빠지거나 YAML 들여쓰기가 틀림). "컵을 그릇에"는 한 번도 없음 |

**결론과 바꾼 점**

- `run_local_pipeline.sh`는 이제 Pipeline A에 `qwen3.5:9b`를, B와 C에는 `qwen3-vl:8b-instruct`를 줍니다(C는 VLM을 부르지 않음). 둘 다 `SIMFOUNDRY_OLLAMA_THINK=0`입니다.
- `run_auto_bg.sh`도 A와 같은 모델을 줍니다. 1단계는 주어진 물체 목록을 쓰므로 VLM을 부르지 않습니다. 그 목록을 계속 쓰는 이유는 qwen3.5도 먼 시점 프레임 2장에서 마커를 놓쳤기 때문입니다.
- `rerun_v2.sh`는 qwen3-vl로 고정했습니다. 응답 캐시도 qwen3-vl의 답을 담고 있습니다.
- shim의 기본값(`qwen3-vl:8b-instruct`, think 미설정)은 그대로입니다.
- 모델을 직접 고르려면 `SIMFOUNDRY_OLLAMA_MODEL`을 줍니다. 래퍼는 시작할 때 Ollama에 그 모델이 있는지 확인합니다.

**한계**

- **비교할 때는 ComfyUI가 이 머신에 없었습니다.** 설치 폴더와 Qwen-Image 가중치가 없어서, 새 기본값으로 A/B 전체를 끝까지 돌리지는 못했습니다. 위 비교는 이미지 모델이 필요 없는 VLM 호출을 모두 재생한 것입니다. ComfyUI를 다시 설치하고 끝까지 돌린 결과는 아래에 있습니다.
- 같은 입력에 온도 0이어도, 다른 작업이 GPU를 같이 쓰면 Ollama가 모델을 작은 버퍼로 올립니다(5,591 → 5,519 MiB). 그러면 경계에 있는 답이 바뀝니다(B1·B7에서 관찰). 그래서 스크립트는 호출마다 GPU가 빌 때까지 기다리고, 겹친 호출은 표시합니다. 위 표는 모두 GPU가 비었을 때의 값입니다.
- qwen3.5의 Modelfile에는 `presence_penalty 1.5`가 들어 있습니다(shim은 온도와 seed만 바꿈). 0으로 바꾼 시험은 다른 작업과 겹쳐 결론을 내지 못했습니다. B1의 중국어 답과 그릇 질량은 0에서도 그대로였습니다.

```bash
repro/vlm_compare/run.sh results/diagnostics/vlm_compare qwen3-vl:8b-instruct qwen3.5:9b:0   # 모델당 약 10분
python repro/vlm_compare/coord_test.py results/diagnostics/vlm_compare/coord                   # 좌표 순서
```

**새 기본값으로 끝까지 실행 (2026-10-16)**

ComfyUI를 다시 설치하고(v0.41.0과 Qwen-Image-2.1-Turbo 가중치. 설치 위치는 `hanbin5/local`의 `local/README.md`) PutMarkerInCup을 A → C → B 순서로 처음부터 돌렸습니다.

- 구성은 VLM만 빼고 11장 v2와 같습니다. 이미지 모델은 `SIMFOUNDRY_IMAGE_BACKEND=qwen21`, A는 `--env-da3 simfoundry --no-stream`과 `s7_mesh.low_vram=true s5_scene.pda_geometric_backend=depth_pro`, B는 `rerun_v2.sh`와 같은 오버라이드입니다.
- `SIMFOUNDRY_OLLAMA_MODEL`은 주지 않았습니다. 그래서 래퍼가 A에는 `qwen3.5:9b`(thinking 끔)를, B에는 `qwen3-vl:8b-instruct`를 줬습니다.
- 응답 캐시는 새 폴더(`model_calls_q35`)를 썼습니다. 출력은 `Data/put_marker_in_cup_q35`, 로그는 `${SIMFOUNDRY_LOG_DIR:-~/simfoundry_logs}/q35_*.log`입니다.

| | v2 (11장, 모두 qwen3-vl) | **q35 (A는 qwen3.5:9b)** |
|---|---|---|
| A 단계 시간 합 | 28분 27초 | 29분 6초 |
| 3단계 프레임 선택 | 47 s. VLM이 프레임 4(후보 2번)를 고름. 화면에서 마커가 컵 테두리에 거의 붙어 있음 | 49 s. VLM이 **프레임 5**(후보 1번)를 고름. 휴리스틱 점수도 1위(0.819, 프레임 4는 0.683)이고 세 물체가 떨어져 보임. 671 토큰, 21 s |
| 5단계 분해 | 8분 51초. 3/3 검출. 가위는 mask 3회 거부 뒤 outline, 컵·마커는 mask | 8분 38초. 3/3 검출(컵 이름은 `white cup`). 제거 방식은 같고 순서만 컵 → 가위 → 마커 |
| 6 / 7단계 | 3분 13초 / 10분 18초 | 3분 41초 / 10분 21초 |
| 8단계 | 1분 49초. 정면 선택 3회 53 s | **1분 14초**. 3회 19 s(이미지 9장이 약 2,200 토큰으로 qwen3-vl의 1/4.5). 컵은 `NONE` |
| 11단계 | 56 s. 질량 호출 3회 13 s | 1분 48초. 3회 62 s(chain of thought 409–1,590 토큰) |
| 질량 컵 · 가위 · 마커 | 150 · 50 · 37.6 g | **25 · 45 · 20 g** (실물 컵 약 10 g, 마커 약 15–20 g) |
| 마찰 컵 · 가위 · 마커 | 0.5 · 0.4 · 0.5 | 0.4 · 0.45 · 0.4 |
| A의 VLM 호출 | 9회 102 s | 9회 119 s |
| MAX_TOKENS · 파싱 실패 | 0 · 0 | 0 · 0 |
| A의 이미지 호출 | 10회, 평균 50.4 s | 10회, 평균 48.6 s |
| C 스모크 테스트 | 통과 (48 s) | **통과** (49 s) |
| B (둘 다 qwen3-vl) | 1단계 4분(첫 시도는 판정 파싱 오류로 실패, 11장), 3단계 7분 8초, 4–7단계 3분 24초 | 1단계 3분 58초(판정 2개 모두 첫 시도에 통과), 3단계 7분 20초, 4–7단계 3분 37초. 부품 이름은 영어, MAX_TOKENS·파싱 실패 0 |
| 제안된 태스크 | `pick_marker_and_place_in_cup`(영상의 실제 과제), `pick_scissors_and_place_above_cup` | `pick_up_marker`, `place_scissors_in_cup`. YAML 2개 모두 파싱됨 |

- **재생 비교와 같은 결과가 실제 실행에서도 나왔습니다.** 3단계는 토큰 한도에도 반복 루프에도 걸리지 않았습니다. 8단계 정면 선택은 호출당 17–19 s에서 5–8 s로 줄었고, 질량은 실물에 가까워졌습니다. 대신 11단계가 chain of thought 때문에 약 50 s 늘어서, A 전체 시간은 v2와 비슷합니다(+39 s). 이 장면에는 그릇이 없어서, 재생에서 본 그릇 5.51 kg 문제는 확인하지 못했습니다.
- **B7 태스크는 두 실행 모두 qwen3-vl의 답입니다**(온도 0.7). 이번에는 영상의 과제(마커를 컵에)가 나오지 않았습니다. A 결과(물체 이름과 포즈)가 달라서인지 샘플링 차이인지는 한 번의 실행으로 구분할 수 없어서, 아래에서 seed를 바꿔 재생했습니다.
- **GPU 충돌로 두 번 멈췄습니다.** 다른 세션의 Isaac Sim 렌더(`repro/g4/render_bg_views.py`, 2.4 GB)가 Hunyuan(8–9 GB, 최대 11.4 GB)과 겹쳐 CUDA OOM이 났습니다. A 7단계(23:32)와 B 3단계(23:59)였습니다.
  - 그 세션의 유휴 판정(`memory.used < 1500 MiB`가 60초)이 단계 사이 모델 로딩 구간을 유휴로 봤습니다.
  - GPU가 빈 뒤 A는 `--skip-successful`로 7단계부터, B는 `--include 3,4,5,6,7`로 다시 돌렸고, 앞 단계 결과는 그대로 썼습니다. 표의 A 7단계와 B 3단계 시간은 다시 돌린 것입니다.
  - 기다린 시간과 날린 단계를 포함한 벽시계 시간은 A 42분, B 22분입니다.
- VRAM: Ollama `qwen3.5:9b` 6.7 GB, `qwen3-vl` 7.8 GB, ComfyUI 편집 최대 10.3 GB입니다.
- 이번 실행은 이 기록이 있는 브랜치의 워크트리에서 돌렸습니다. `Data`·`deps`·`checkpoints`·`assets`를 실행용 체크아웃에 링크하고, `--root-dir`로 실행용 체크아웃의 `Data`를 줘서 장면 JSON에 기록되는 경로가 v2와 같습니다. 실행용 체크아웃에서는 다음과 같습니다.

```bash
export SIMFOUNDRY_IMAGE_BACKEND=qwen21   # SIMFOUNDRY_OLLAMA_MODEL은 주지 않음
C=(--cache-mode --model-cache-dir .cache/simfoundry/model_calls_q35)
repro/run_local_pipeline.sh A_reconstruction --scene-name put_marker_in_cup_q35 \
  --video-fpath "$PWD/docs/assets/example_videos/PutMarkerInCup.mp4" --env-da3 simfoundry --no-stream "${C[@]}" -- \
  s7_mesh.low_vram=true s5_scene.pda_geometric_backend=depth_pro
repro/run_local_pipeline.sh C_application --scene-name put_marker_in_cup_q35 --mode smoke-random
repro/run_local_pipeline.sh B_augmentation --scene-name put_marker_in_cup_q35 "${C[@]}" -- \
  prompt_cousin_structured.max_objects=2 prompt_cousin_structured.max_generated_images_per_object=1 cousin_generation.low_vram=true
```

**B7 태스크 제안: A의 출력 탓인가, 샘플링 탓인가 (2026-10-17)**

B7은 온도 0.7로 샘플링하지만, 스테이지가 seed를 넘기지 않아 매번 seed 0으로 부릅니다. 장면이 같으면 다시 돌려도 답이 같으므로, 한 번의 실행으로는 장면 탓인지 뽑기 탓인지 가를 수 없습니다. 그래서 두 장면의 B7 호출을 seed만 0–29로 바꿔 30번씩 다시 보냈습니다(`repro/vlm_compare/replay_b7_seeds.py`, 64호출에 8분). 결과는 `results/diagnostics/vlm_compare/b7_q35/`에 있습니다.

- `replay_other_calls.run_b7`을 그대로 씁니다. 스테이지의 프롬프트 빌더와 파서, `Gemini(...)` → shim → Ollama 경로, `qwen3-vl:8b-instruct`, 온도 0.7, keep_alive 0이고, 응답 캐시는 끕니다.
- 태스크는 Pipeline B처럼 2개를 요청합니다(`run.sh`의 `propose_scene_task.num_tasks=2`). 위 비교 표의 B7 재생은 설정 파일 기본값인 5개를 요청했습니다.
- seed 0이 곧 파이프라인의 호출입니다. 두 장면 모두 실행이 쓴 태스크 YAML과 같은 답이 나왔고, 토큰 수도 로그와 같습니다(프롬프트/생성 2133/180, 2130/158). 같은 seed를 다시 보내도 답이 같습니다(4/4).
- 호출마다 GPU가 빌 때까지 기다리고, 다른 작업과 겹친 호출은 버린 뒤 다시 보냅니다(이번에는 0회).
  - B가 띄워 둔 유휴 ComfyUI(모델 내림, 큐 비어 있음)는 작업으로 치지 않습니다. 파이프라인의 B7도 그 옆에서 돌았습니다.
  - `bench_common`의 판정은 Ollama가 모델을 올리고 내릴 때 잠깐 생기는 자식 프로세스도 다른 작업으로 셉니다. 장치를 확인하는 `llama-server --list-devices`와 `ollama gpu-discover`(좀비 상태), 내려가는 러너가 그렇습니다. NVML에는 보이는데 명령줄이 비어 있기 때문입니다. seed가 같으면 답도 걸린 시간도 같아서, 같은 호출이 거듭 버려졌습니다. 이 스크립트는 부모가 Ollama인 프로세스를 빼고 셉니다.

| seed 0–29, 장면마다 30회 | v2 (A도 qwen3-vl) | q35 (A는 qwen3.5:9b) |
|---|---|---|
| 프롬프트의 물체 목록 | iter_0 가위, iter_1 `white_cup_with_green_pattern`, iter_2 마커 | iter_0 `white_cup`, iter_1 가위, iter_2 마커 |
| **마커를 컵에 (InsideAABB)** | **21/30** (seed 0 포함) | **6/30** (seed 0은 빠짐) |
| 파싱된 태스크 | 30회 모두 2개 | 30회 모두 2개 |
| 로드되지 않는 태스크 (술어의 그룹이 매핑 키가 아님) | 0/60 | 4/60 (마커를 컵에 1개 포함) |
| 가장 흔한 두 답 (30회 중) | 마커를 컵에 + 가위를 컵 위에 띄우기 14, 가위를 컵에 + 마커 들기 7 | 가위를 컵에 + 마커 들기 13(실행의 답과 같은 꼴), 마커를 컵에 + 가위 들기 5 |

**주된 원인은 샘플링이 아니라 A의 출력입니다.** 모델·온도·seed가 같아도 v2 장면은 30회 중 21회(70%, 95% 구간 52–83%) 마커를 컵에 넣는 태스크를 냈고, q35 장면은 6회(20%, 10–37%)뿐입니다(Fisher 정확 검정 p = 0.0002). 파이프라인의 seed 0은 두 장면 모두에서 흔한 쪽이었습니다. v2에서는 열에 일곱 번 그 태스크가 나오고, q35에서는 다섯에 네 번 나오지 않습니다. 두 호출은 B7이 받는 입력만 다르고, 그 입력은 Pipeline A가 만듭니다. 물체 목록에서는 5단계가 컵을 `white cup`으로 짧게 부르고 컵 → 가위 → 마커 순서로 제거해서, 컵과 가위의 ID가 서로 바뀌었습니다. 장면 이미지에서는 포즈가 다릅니다. 두 장면 모두 한 물체는 컵에 넣고 다른 하나는 들거나 컵 위에 두는 답이 대부분인데, q35에서는 컵에 넣는 쪽이 마커보다 가위인 경우가 많습니다. 두 이미지는 배치가 비슷해 보여(화면에서 마커가 컵 바로 위, 가위가 오른쪽 아래) 물체 목록 쪽이 의심되지만, 이 재생으로는 둘을 가를 수 없습니다.

```bash
SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_OLLAMA_MODEL=qwen3-vl:8b-instruct SIMFOUNDRY_OLLAMA_THINK=0 SIMFOUNDRY_OLLAMA_KEEP_ALIVE=0 \
  GCLOUD_PROJECT=local python repro/vlm_compare/replay_b7_seeds.py results/diagnostics/vlm_compare/b7_q35 --seeds 30
```

## 17. 96 GB GPU(Colab G4)에서 저장소 기본값으로 다시 돌리기 (2026-10-16)

12 GB 때문에 바꾸거나 하지 못한 실행을 Colab G4 런타임(RTX PRO 6000 Blackwell Server Edition 96 GB)에서 저장소 기본값대로 다시 돌렸습니다. 스크립트는 `repro/g4/`에 있습니다.

### 17.1 3060 실행과 다른 점

v2(11장) 구성에서 12 GB 때문에 바꾼 것만 되돌렸습니다.

| 항목 | RTX 3060 12 GB (11·13장) | G4 96 GB (이 장) |
|---|---|---|
| A 5단계 PDA 기하 깊이 백엔드 | `depth_pro` | `da3` (기본값) |
| A 7단계 Hunyuan3D-2.1 | `low_vram=true` | `false` (기본값) |
| A 5~8단계 | 순차 (`--no-stream`) | 스트리밍 (기본값) |
| B cousin 메시 | `low_vram=true` | `false` (기본값) |
| 로컬 모델 | 호출마다 VRAM에서 내림 | 계속 올려 둠 (`SIMFOUNDRY_OLLAMA_KEEP_ALIVE=24h`, `SIMFOUNDRY_COMFY_FREE_AFTER=0`, `SIMFOUNDRY_LOCAL_OFFLOAD=0`) |
| auto_bg 프레임 수 | 160 | 400 (기본값) |
| VOID | `VOID_LOW_VRAM=1` (타일 디코딩, CPU 오프로딩) | 원래 코드 경로 |
| void DA3 | 앵커 정렬 (`repro/da3_anchored_stage.py`) | 저장소 2b (220프레임 덩어리 연쇄) |
| auto_bg 장면 | PutMarkerInCup | PutMarkerInCup, PutCupInBowl |

그대로 둔 것도 있습니다.
- Gemini 대신 로컬 모델: Ollama `qwen3-vl:8b-instruct`, ComfyUI Qwen-Image-2.1-Turbo Q8_0. 16장 이후 `run_local_pipeline.sh`는 Pipeline A에 `qwen3.5:9b`를 쓰므로, `rerun_g4.sh`와 `auto_bg_g4.sh`는 `qwen3-vl:8b-instruct`로 고정합니다.
- 원본 SAM3, 물체 제거 기본 순서 `[mask, outline, bbox]`.
- DA3를 simfoundry 환경에서 실행.
- auto_bg 1단계의 물체 목록을 재구성 5단계에서 가져오기(로컬 VLM 보완, 13.2).

### 17.2 G4 런타임 준비

- **임대**: Colab CLI 0.7.4로 `colab new --gpu G4`를 썼습니다. 런타임 한 대에 시간당 8.9 compute unit입니다. 명령은 `colab ssh --proxy-mode`를 SSH의 ProxyCommand로 써서 보냈습니다. 유료 플랜에서는 SSH가 허용됩니다. 런타임 하나에 SSH 연결은 하나만 열립니다(두 번째는 HTTP 429).
- **사양**: GPU 96 GB(sm_120, 드라이버 580.82.07), 48 vCPU, RAM 176 GB, 디스크 236 GB, Ubuntu 24.04.
- **설치**: `repro/g4/setup_g4.sh`로 했습니다. 저장소 설치 스크립트는 모두 cu128과 sm_120을 지원해서 그대로 썼고, 아래 네 가지만 더했습니다.
  - **Vulkan**: Colab은 드라이버의 CUDA·EGL·GLX 라이브러리는 마운트하지만 Vulkan ICD 파일이 없어서, Isaac Sim이 GPU를 찾지 못합니다. 같은 버전(580.82.07) NVIDIA 드라이버 패키지에서 빠진 사용자 공간 라이브러리와 ICD 파일만 꺼내 넣었습니다. 커널 모듈은 건드리지 않았습니다.
  - **libGLU**: Colab 이미지에 없어서 Iray/MDL 라이브러리(libneuray)가 로드되지 않았습니다. 그러면 재질 셰이더를 만들지 못해 RTX 렌더러가 시작되지 않습니다(13단계에서 크래시). `libglu1-mesa`를 설치했습니다. 다른 시스템 라이브러리는 빠진 것이 없음을 ldd로 확인했습니다.
  - **패키지 버전 고정**: 3060 환경을 만든 10-08 이후에 opencv 5.0과 pyarrow 26이 나왔고, 둘 다 numpy 2 이상을 요구합니다. simfoundry 환경은 numpy 1.26.4를 고정하므로 `import omnigibson`이 실패했습니다. `repro/g4/constraints_simfoundry.txt`로 3060 환경과 같은 버전에 고정했습니다(opencv 4.11.0.86, pyarrow 25.0.1, cmeel-boost 1.83.0, pin 2.7.0, transformers 5.19.0 등). 다른 환경은 설치 스크립트가 스스로 버전을 고정합니다. 모든 환경에 걸면 hunyuan의 opencv 고정(4.10)과 부딪힙니다.
  - **SAM 2**: auto_bg 1단계는 simfoundry 환경에서 `sam2`를 import합니다. 그런데 저장소 설치 스크립트는 sam2를 any6d 환경에만 설치합니다. 3060 환경에는 따로 설치돼 있었습니다. 같은 커밋(`2b90b9f`)을 설치했고, `install_simfoundry.sh`에도 추가했습니다.
- **런타임 두 대**: 환경 전체, VOID 가중치(41 GB), 모델 캐시를 합치면 236 GB에 들어가지 않습니다.
  - 런타임 1: A·B·C를 맡았습니다. 끝난 뒤 ComfyUI 모델을 지우고 void·nerfstudio를 설치해 PutCupInBowl의 splat-prep A와 auto_bg도 맡았습니다.
  - 런타임 2: PutMarkerInCup의 splat-prep A와 auto_bg를 맡았습니다.
  - 7단계(3dgrut로 PLY → USDZ, 43초)와 배경 위 조작 시연은 3dgrut 환경이 있는 3060 컴퓨터에서 했습니다.
- **비용**: 런타임 1은 3.4시간, 런타임 2는 2.1시간을 썼습니다. 합계 약 48 compute unit입니다(계정 잔액은 1,259.99에서 1,208.73으로 줄었고, 차이 중 약 3은 이 작업과 무관하게 먼저 켜져 있던 G4 런타임 몫입니다). 끝난 뒤 두 런타임을 모두 종료했습니다.

### 17.3 결과: A·B·C (시간을 잰 실행)

워밍업으로 A를 한 번 돌려 모델 다운로드와 셰이더 컴파일을 끝낸 뒤 쟀습니다(`repro/g4/rerun_g4.sh warmup|timed`). 로컬 모델의 응답 캐시는 새 폴더를 썼습니다. cache 모드는 캐시에 있는 응답을 다시 쓰기 때문입니다.

| 단계 (초) | PutMarkerInCup 3060 v2 | PutMarkerInCup G4 | PutCupInBowl 3060 v2 | PutCupInBowl G4 |
|---|---|---|---|---|
| 1b~4 | 75 | 44 | 77 | 45 |
| 5 장면 분해 | 531 | 84 | 551 | 79 |
| 6 업샘플 | 193 | 66 | 191 | 65 |
| 7 메시 | 618 | 216 | 635 | 232 |
| 8 포즈 | 109 | 71 | 112 | 71 |
| 5~8 실제 경과 | 1,450 (순차) | **318** (스트리밍) | 1,489 (순차) | **325** (스트리밍) |
| 10~12 | 63 | 62 | 84 | 78 |
| 13~14 (Isaac Sim) | 119 | 97 | 98 | 97 |
| **A 전체** | **1,708 (28.5분)** | **524 (8.7분)** | **1,748 (29.1분)** | **548 (9.1분)** |

- **속도**: 물체 3개 장면이 8.7~9.1분으로, 물체당 약 3분입니다. 논문 Table L.4는 RTX 3090과 Gemini API로 물체 2개 장면에 19.98분이고, FAQ는 물체당 약 5분입니다. 3060보다 3.2배 빠르고, 대부분은 5~8단계에서 줄었습니다.
  - 3060에서는 로컬 모델을 호출마다 VRAM에 다시 올리고, 단계를 하나씩 돌렸습니다.
  - G4에서는 이미지 편집 한 번이 평균 51.7초(10장)에서 5.9초로 줄었고, VLM 호출은 중앙값 1.2초입니다.
  - 5~8단계가 동시에 돌 때 GPU 사용량은 최대 59 GB였습니다(상주한 로컬 모델 약 35 GB 포함). VRAM 예산 때문에 기다린 단계는 없었습니다.
- **C 스모크 테스트**: 두 장면 모두 통과했습니다(53초, 49초).
- **B (PutMarkerInCup)**: 321초가 걸렸습니다(3060 v2는 662초, 1단계 첫 시도 240초는 따로). 3단계 cousin 메시는 428초에서 122초로 줄었습니다.
  - 장면 변형은 2개 모두 유효했습니다.
  - 컵 cousin은 이번에도 "같은 바닥 지름에서 더 긴 컵"이었습니다.
  - 제안된 태스크는 `pick_marker_and_place_in_cup`과 `pick_scissors_and_lift`(Lifted ≥ 5 cm)입니다. 마커 태스크의 목표 조건이 v2의 `InsideAABB`가 아니라 `OnTopAABB`로 나왔습니다. 같은 8B VLM이라도 실행마다 답이 달라집니다.
- **품질**: 물체 구성은 3060과 같고, 크기와 정합 잔차도 같은 수준입니다(`repro/g4/compare_runs.py`).

| 장면 | 물체 | 크기 (cm) 3060 v2 → G4 | 정합 잔차 평균 3060 v2 → G4 |
|---|---|---|---|
| PutMarkerInCup | 컵 | 9.5×9.5×11.2 → 9.9×9.7×11.3 | 1.97 → 2.74 mm |
| | 마커 | 길이 13.7 → 13.3 | 3.79 → 2.90 mm |
| | 가위 | 길이 20.0 → 21.3* | 1.88 → 2.13 mm |
| PutCupInBowl | 그릇 | 25.7×25.8×13.7 → 26.0×26.3×12.3 | 12.00 → 13.28 mm |
| | 펜 | 길이 12.8 → 12.9 | 2.85 → 5.02 mm |
| | 컵 | 9.4×9.1×11.0 → 9.4×8.9×10.8 | 2.87 → 2.93 mm |

\* G4의 가위는 정면 정렬이 45° 돌아가 축 정렬 bbox가 15.3×16.0 cm로 나옵니다. 그래서 회전 bbox의 긴 변을 적었습니다(3060 v2는 같은 방식으로 20.6 cm).

장면 평균 잔차는 PutMarkerInCup 2.55 → 2.59 mm, PutCupInBowl 5.91 → 7.08 mm입니다. 펜은 실행마다 크게 흔들립니다(v1 6.0, v2 2.85, G4 5.02 mm).

VLM이 추정한 질량도 실행마다 다릅니다. 가위는 50 g에서 150 g, 펜은 10 g에서 15 g, 컵은 70 g에서 80 g이 됐습니다.

### 17.4 결과: 자동 3DGS 배경 (저장소 기본값)

`repro/g4/auto_bg_g4.sh`로 splat-prep A(400프레임, 672×384, auto_bg README의 명령)를 돌리고, 이어서 `repro/run_auto_bg.sh`(`VOID_LOW_VRAM=0 VOID_DA3=canonical`)를 돌렸습니다.

| 단계 | 3060 PutMarkerInCup (160프레임) | G4 PutMarkerInCup (400) | G4 PutCupInBowl (400) |
|---|---|---|---|
| splat-prep A | | 18.6분** | 11.5분 |
| 1 quadmask | 1.9분 | 0.9분 | 0.9분 |
| 2 VOID pass1 | 26.5분 | 15.5분 | 15.2분 |
| 3 VOID pass2 | 43분 | 12.3분 | 12.3분 |
| void DA3 | 1.3분 (앵커) | 1.0분 (2b, 220프레임 덩어리 2개) | 1.0분 |
| 5 splat 80k | 43분 | 20.0분 | 18.7분 |
| 1~6 합계 | 약 116분 | 49.8분 | 48.2분 |

\** 런타임 2에서는 첫 실행이라 모델 다운로드가 포함됐습니다.

품질은 시점별로 나눠 봤습니다.

**촬영 시점 근처**에서는 splat이 VOID 결과를 재현합니다.
- 학습 프레임에서 nerfstudio로 렌더하면 세 splat 모두 VOID 프레임과 거의 같습니다.
- 학습에서 뺀 10% 프레임으로 `ns-eval`을 돌린 수치는 아래 표에 있습니다(`repro/g4/eval_splat.sh`).
- 다만 이 프레임들은 학습 궤적 사이에 끼어 있어서 새 시점의 품질은 재지 못합니다. PSNR 22~23 dB도 정적 장면 splat으로는 낮은 값입니다.

**로봇 카메라 시점**에서는 세 배경 모두 쓸 수 없습니다.
- 조작 시연을 로봇의 외부 카메라 두 대(`sim_franka_teleop`)로 렌더하면 책상과 벽이 긴 줄무늬와 무지개색 부유물로 무너집니다. 촬영되지 않은 방향은 하늘(skybox)이 보이고, 물체만 제대로 보입니다.
- 13장의 3060 배경도 같습니다. 13장의 "실제 프레임과 겹친다"는 촬영 카메라 시점에서만 확인한 것이었습니다.
- **원인**: 입력은 서서 책상 주위를 한 번 훑은 14.9초 휴대폰 영상뿐입니다. 로봇 외부 카메라는 낮은 높이에서 옆으로 보는데, 이 시점은 학습 시점에서 멉니다.
  - 그래서 학습 시점에서만 맞고 3D 모양은 틀린 가우시안이 많습니다. 세 축 길이 비가 중앙값 84~283배이고, 상위 1%는 10⁶배인 바늘·원판 모양입니다. 이런 가우시안이 다른 시점에서 줄무늬로 보입니다.
  - splatfacto의 scale regularization은 저장소 기본값에서 꺼져 있습니다(`use_scale_regularization=False`).
- **12 GB 때문이 아님**: 프레임 수(160 → 400)와 VOID·DA3 경로를 저장소 기본값으로 되돌려도 이 한계는 그대로입니다.
- **물리에는 영향 없음**: 배경은 보이기만 하는 물체라 조작 시연은 세 배경 모두 성공했습니다. 하지만 시점이 바뀌면 카메라 영상이 망가지므로, 이 배경에서 카메라 입력을 쓰는 정책을 평가하거나 학습하기는 어렵습니다.

아래 표의 측정 도구는 다음과 같습니다. 3060 결과도 같은 스크립트로 다시 쟀고, 3060의 궤적 수치는 13.1의 기록과 같습니다.
- void-DA3 궤적: `repro/g4/compare_da3_tracks.py`로 원본 DA3 카메라 궤적과 비교했습니다. 4단계와 같은 강체 정렬입니다.
- 깊이 축척: `repro/g4/splat_depth_scale.py`로 splat이 렌더한 깊이를 두 DA3 깊이와 비교했습니다.

| | 3060 PutMarkerInCup | G4 PutMarkerInCup | G4 PutCupInBowl |
|---|---|---|---|
| PSNR (dB) / SSIM / LPIPS | 23.55±3.87 / 0.669 / 0.087 (16프레임) | 23.29±1.69 / 0.671 / 0.084 (40프레임) | 22.04±0.92 / 0.650 / 0.094 (40프레임) |
| 가우시안 수 | 388k | 390k | 382k |
| 로봇 카메라 시점 | 무너짐 | 무너짐 | 무너짐 |
| void-DA3 궤적 오차 (강체, 평균/최대) | 4.6 / 12.1 mm, 방향 0.87° | 8.1 / 30.4 mm, 방향 3.4° | 25.7 / 50.7 mm, 방향 5.6° |
| void-DA3 축척 (원본 대비) | 0.990 | 0.991 | **1.062** |
| 렌더 깊이 / 원본 DA3 (중앙값) | 1.012 | 1.020 | **0.937** |
| 렌더 깊이 / void-DA3 (중앙값) | 0.997 | 0.992 | 0.997 |
| 조작 시연 (`InsideAABB`) | 성공, 864스텝 | 성공, 857스텝 | 성공, 699스텝 |

- **void-DA3**: 저장소 2b 경로는 96 GB에서 설계대로 돌았습니다. 220프레임 덩어리 2개였고, 12 GB의 16프레임 덩어리처럼 20° 넘게 틀어지지 않았습니다.
- **축척 문제 (PutCupInBowl)**: void-DA3와 원본 DA3의 축척이 6.2% 달랐습니다.
  - 저장소 4단계는 두 DA3를 강체로만 맞추고, 5단계는 원본 DA3의 카메라와 void-DA3의 깊이로 학습합니다. 그래서 splat 깊이가 물체가 놓인 원본 DA3 세계보다 6.3% 얕습니다.
  - 3060 실행과 PutMarkerInCup에서는 두 DA3 축척 차이가 1% 안쪽이었습니다.
  - 광선 방향의 깊이 오차라 촬영 카메라 시점에서는 거의 보이지 않습니다.
- **영상과 그림**: 3060 컴퓨터의 `~/simfoundry_logs/g4/local/`에 있습니다.
  - `sim_gsbg_*_g4.mp4`: 촬영 시점, 90° 렌즈.
  - `sim_gsbg_*_cams.mp4`: 로봇 카메라 시점.
  - `figures/`: 학습 프레임 렌더 대 VOID 프레임(`*_train.jpg`), 로봇 카메라 첫 프레임(`cams_*.jpg`), 입력 대 VOID(`bowl_real_vs_void.jpg`).

### 17.5 정리

- **A·B·C**: 12 GB 조정을 모두 뺀 저장소 기본값(96 GB)으로도 결과가 같은 수준이었습니다. 물체 구성, 크기, 정합 잔차, C 스모크 테스트, B의 장면 변형이 모두 같았습니다. 12 GB 조정은 속도만 늦췄습니다.
- **복원 시간**: 물체 3개 장면이 9분 안팎, 물체당 약 3분입니다. 논문의 물체당 약 5분(RTX 3090, Gemini)보다 빠릅니다.
- **자동 배경**: 저장소 기본값으로 두 장면 모두 끝까지 돌았습니다. 하지만 결과 배경은 촬영 시점 근처에서만 맞고, 로봇 카메라 시점에서는 쓸 수 없습니다. 3060 배경도 같으며, 13장의 평가는 이 점을 놓쳤습니다.
- **배경 재학습 (17.7)**: 설정을 바꿔 다시 학습하고 빈 공간의 가우시안을 지우니, 로봇 카메라 시점에서도 책상과 물체가 보이게 됐습니다. 하지만 영상에 없던 영역은 학습으로 채워지지 않습니다. 남은 방법은 다음과 같습니다(미수행).
  - 로봇 카메라 높이와 방향을 덮는 영상을 새로 찍기.
  - 시뮬레이션 카메라를 촬영 궤적 근처에 두기.
  - 로봇 시점에는 저자 공개 메시 배경(12장)을 쓰기.
- **축척 문제**: void-DA3와 원본 DA3의 축척이 다르면 배경 깊이가 어긋납니다(PutCupInBowl 6%). 17.7에서는 void-DA3를 닮음 변환으로 원본 축척에 맞춘 뒤 다시 학습했습니다(`repro/g4/scale_void_da3.py`). 저장소 4·5단계에는 아직 반영하지 않았습니다.
- **저장소 설치 스크립트의 빈틈 두 가지**
  - simfoundry 환경에 sam2가 없습니다. `install_simfoundry.sh`에 추가했습니다.
  - Colab 같은 컨테이너에는 Vulkan ICD와 libGLU가 없습니다. `setup_g4.sh`에서 보완했습니다.

### 17.6 다시 실행하는 법

```bash
colab new -s sf-g4 --gpu G4      # 런타임 하나 (A·B·C), 둘째는 -s sf-g4b
# 체크아웃을 /content/SimFoundry에 복사하고, HF 토큰(facebook/sam3 게이트)과
# FoundationPose 가중치(/content/fallback/deps/FoundationPose/weights)를 올린 뒤:
G4_ENVS="" bash repro/g4/setup_g4.sh all                   # A·B·C 런타임
G4_ENVS="void nerfstudio" bash repro/g4/setup_g4.sh all    # auto_bg 런타임
bash repro/g4/rerun_g4.sh warmup && bash repro/g4/rerun_g4.sh timed
AUTO_BG_STOP_BEFORE_ASSETS=1 bash repro/g4/auto_bg_g4.sh   # 7단계는 3dgrut 환경이 있는 곳에서
```

결과 비교(3060 컴퓨터, simfoundry 환경):

```bash
python repro/g4/compare_runs.py 3060v2=Data/put_marker_in_cup_v2 G4=Data/put_marker_in_cup_g4
python repro/g4/compare_da3_tracks.py Data/put_cup_in_bowl_bg_g4
repro/g4/ns_run.sh python repro/g4/splat_depth_scale.py Data/put_cup_in_bowl_bg_g4
repro/g4/eval_splat.sh Data/put_cup_in_bowl_bg_g4/auto_bg/splat eval.json
```

결과 위치: 장면은 3060 컴퓨터의 `Data/{put_marker_in_cup,put_cup_in_bowl}{_g4,_bg_g4}`, 로그는 `~/simfoundry_logs/g4/`, 응답 캐시는 `.cache/simfoundry/model_calls_g4*`에 있습니다. Drive에는 아직 올리지 않았습니다.

### 17.7 배경 재학습: 로봇 카메라 시점에서 확인하며 (2026-10-17)

17.4의 배경은 로봇 카메라 시점에서 무너졌습니다. 그래서 학습 설정을 바꾸고 후처리를 더해 다시 만들고, 매번 로봇 카메라 시점으로 렌더해 확인했습니다.

**확인 방법**
- `repro/g4/render_bg_views.py`가 7단계 장면을 실제 카메라와 로봇 외부 카메라 4대(DROID 배치 2대, teleop 배치 2대)로 렌더해 한 장으로 만듭니다. 로봇은 시연처럼 물체 앞 0.55 m에 둡니다.
- 학습 제외 프레임의 PSNR은 학습 궤적 근처 시점만 보므로 참고만 했습니다. 판정은 이 그림으로 했습니다.
- `repro/g4/check_bg_variant.sh`가 변형 하나를 받아 7단계와 렌더까지 합니다. 변형은 `assets/scenes/<scene>_<variant>/`에 따로 만들어서, 기본 배경은 그대로 둡니다.

**학습 환경**
- G4는 계정의 동시 3대 한도가 이 작업과 무관한 G4 세 대로 차 있어서 받지 못했습니다. 대신 A100 80 GB(`--gpu A100 --high-mem`, 시간당 6.77 CU)를 썼습니다. A100에는 RT 코어가 없어 Isaac Sim 렌더는 못 하지만 splat 학습에는 필요 없습니다. 렌더는 3060 컴퓨터에서 했습니다.
- 4~6단계는 nerfstudio 환경에서 돌렸습니다. `simfoundry.utils.processing_utils`가 import하는 `supervision`만 따로 넣으면 됩니다(`setup_g4.sh splat`). bridge 결과는 simfoundry 환경과 1e-9까지 같았고, 설치는 6분으로 줄었습니다.
- 모든 변형은 먼저 void-DA3를 원본 DA3 축척에 맞추고 seed를 다시 만든 뒤 학습했습니다(`scale_void_da3.py`). 닮음 변환 축척은 PutMarkerInCup 0.9907, PutCupInBowl 1.0619였습니다(17.4의 축척 문제).
- 학습 스크립트는 `repro/g4/retrain_bg.sh`입니다. ns-train에 옵션을 더 넘길 수 있게 `5_train_bg_splat.py`에 선택 설정 `s5_train_bg_splat.extra_model_args`를 추가했습니다(기본값은 빈 목록).

**변형**

| 이름 | 바꾼 설정 (나머지는 저장소 기본값) |
|---|---|
| sreg_sh0_depth2_30k | scale regularization(가우시안 축 길이 비 상한 10), SH 차수 3 → 0, 깊이 손실 0.5 → 2.0, 저불투명도 제거 기준 0.005 → 0.1, 30k 반복 |
| iso3_sh0_depth2_30k | 위와 같고 축 길이 비 상한 10 → 3 |
| sreg + crop | sreg 결과에서 책상 면 10 cm 아래(OG z < -0.10 m) 가우시안을 제거(`crop_bg_splat.py`). 19~30% 제거 |
| sreg + carve | sreg 결과에서 빈 공간의 가우시안을 제거(`carve_bg_splat.py`). 학습 프레임 400장의 깊이로 판정해, 관측 표면보다 max(3 cm, 5%) 앞에 있다고 나온 프레임이 보이는 프레임의 20% 이상(3회 이상)인 것과 어느 프레임에도 안 보이는 것을 지움. PutCupInBowl 6%, PutMarkerInCup 17% 제거 |

80k 반복 변형도 띄웠지만, A100에서 6개를 함께 돌리니 반복 한 번에 110 ms라 2.5시간이 걸릴 상황이어서 멈췄습니다. 판정은 30k 반복으로 했습니다.

**결과**

| 변형 | 학습 제외 프레임 PSNR / SSIM / LPIPS, PutCupInBowl | PutMarkerInCup | 로봇 카메라 시점 |
|---|---|---|---|
| 17.4 기본 | 22.04 / 0.650 / 0.094 | 23.29 / 0.671 / 0.084 | 무너짐: 무지개 줄무늬, 바늘 모양, 하늘. 책상과 물체를 알아보기 어려움 |
| sreg_sh0_depth2_30k | 22.19 / 0.646 / 0.101 | 20.86 / 0.634 / 0.132 | 책상이 물체 아래 판으로 보이고, 그릇·컵·마커가 그 위에 보임. 무지개 줄무늬 없음. 카메라 앞 회색 안개, 책상 가장자리 파편, 하늘은 남음 |
| iso3_sh0_depth2_30k | 22.20 / 0.650 / 0.106 | 20.78 / 0.632 / 0.137 | sreg와 거의 같음. 파편이 조금 뭉툭해짐 |
| sreg + crop | (재학습 없음) | | sreg와 거의 같음. 파편은 책상 아래가 아니라 책상 높이와 그 위에 있었음 |
| **sreg + carve** | (재학습 없음) | | **가장 나음.** 카메라 앞 회색 안개와 책상 위 파편이 크게 줄어듦. 하늘과 책상 가장자리 파편은 남음. PutMarkerInCup은 실제 카메라 시점의 왼쪽 아래가 조금 흐려짐 |

- 비교 그림: 3060 컴퓨터의 `~/simfoundry_logs/g4/bgfix/compare_{bowl,marker}.jpg`(행은 변형, 열은 실제 카메라와 로봇 카메라 4대)와 변형별 `views_*.jpg`입니다.
- PutMarkerInCup의 PSNR이 2.4 dB 떨어진 것은 SH 0(시점마다 색을 바꾸지 못함)과 강한 정규화 때문입니다. 실제 카메라 시점의 그림에서는 차이가 거의 보이지 않습니다.

**남은 한계**: 입력은 서서 책상을 내려다본 15초 휴대폰 영상 하나뿐입니다. 로봇 카메라 높이에서 보이는 책상 옆, 책상 아래, 위쪽 공간은 찍히지 않았고, 학습 설정이나 후처리로는 채울 수 없습니다. 그곳은 하늘(skybox)이나 파편으로 보입니다. 로봇 카메라 영상을 쓰는 정책 평가에 쓰려면 17.5의 방법(로봇 시점을 덮는 촬영 등)이 필요합니다.

**진행 중 겪은 문제**
- 처음 빌린 G4 런타임은 3060 컴퓨터가 재부팅되면서 keepalive 연결이 끊긴 뒤 20분 안에 회수됐습니다. 설치와 업로드를 다시 했습니다.
- `colab ssh -s 이름`은 그 이름의 세션이 없으면 CPU 런타임을 새로 만듭니다. G4 할당이 거절된 직후 한 번 생겨서 바로 껐습니다.
- 변형 여러 개를 동시에 띄우면 gsplat이 CUDA 확장을 같은 폴더에 동시에 JIT 빌드하다 서로 깨뜨립니다. `retrain_bg.sh`는 잠금을 건 채 먼저 한 번만 빌드합니다.
- 3060 컴퓨터의 GPU는 다른 세션과 같이 씁니다. 렌더(3~4 GB)가 다른 세션의 파이프라인(Hunyuan 8~9 GB)을 두 번 메모리 부족으로 멈췄습니다. 그 뒤로는 다른 세션의 파이프라인 프로세스가 없을 때만 렌더했습니다.

**비용**: A100 약 1.25시간, 약 8.5 CU입니다.

**다시 실행하는 법** (A100 런타임)

```bash
colab new -s sf-bgfix2 --gpu A100 --high-mem
# 체크아웃을 /content/SimFoundry에, 장면의 학습 입력(s2_da/da/exports/npz, auto_bg/da3/void,
# auto_bg/void/pass2/cleaned_frames, s4_frame, s3_ground)을 Data/<scene>/에 올린 뒤:
bash repro/g4/setup_g4.sh splat
bash repro/g4/retrain_bg.sh put_cup_in_bowl_bg_g4 sreg_sh0_depth2_30k   # Data/<scene>/auto_bg/splat_<variant>/
```

3060 컴퓨터에서(결과 폴더를 받은 뒤):

```bash
python repro/g4/scale_void_da3.py Data/put_cup_in_bowl_bg_g4 Data/put_cup_in_bowl_bg_g4/auto_bg/da3/void_sim3.npz
python repro/g4/carve_bg_splat.py Data/put_cup_in_bowl_bg_g4 \
  Data/put_cup_in_bowl_bg_g4/auto_bg/splat_sreg_sh0_depth2_30k/export/put_cup_in_bowl_bg_g4_bg.ply \
  Data/put_cup_in_bowl_bg_g4/auto_bg/splat_sreg_sh0_depth2_30k_carve/export/put_cup_in_bowl_bg_g4_bg.ply
repro/g4/check_bg_variant.sh put_cup_in_bowl_bg_g4 sreg_sh0_depth2_30k_carve iter_2 iter_0 views.jpg
```

조작 시연에 쓰려면 `repro/scripted_pick_place.py --scene-file assets/scenes/<scene>_<variant>/<scene>_<variant>_scene_state_auto_bg.json`으로 엽니다.

## 18. Table G.1의 시뮬레이션 쪽: 저자 장면 5개에서 공개 π0·π0.5-DROID 평가 (2026-10-16~17)

Table G.1은 같은 정책을 실제 로봇과 SimFoundry 장면에서 평가한 성공률을 나란히 놓은 표입니다. 이 장에서는 그중 시뮬레이션 쪽을 다시 돌렸습니다. 정책은 공개된 π0-DROID와 π0.5-DROID(openpi 관절 위치 체크포인트)이고, 장면은 저자가 공개한 DROID 장면 5개, 평가 코드는 저장소의 평가 단계입니다. 과제·정책마다 25에피소드씩 돌렸고, 스크립트는 `repro/policy_eval/`에 있습니다. 19장은 여기에 우리 복원 장면과 GR00T N1.6을 더합니다.

**무엇과 비교할 수 있나.** 논문 5.1절과 부록 H에 따르면 Table G.1의 행은 두 종류입니다.
- **Marker in Cup, Serve Fruits**: 사전학습 정책을 미세조정 없이 평가한 행입니다. 공개 체크포인트와 같은 정책이므로 **직접 비교**할 수 있습니다. 같은 종류의 Cup in Bowl·Clear Table은 저자 장면이 공개되지 않았습니다(19장).
- **Stack Dishware, Store Marker, Throw Away Trash**: 실제 시연 50개로 **미세조정한** 정책의 행입니다. 그 체크포인트는 공개되지 않았으므로, 여기서 돌린 공개 체크포인트의 수치는 Table G.1과 다른 정책의 결과입니다. 대신 논문 FAQ 6이 공개 체크포인트의 성공률을 적어 두었습니다(Store Marker·Throw Away Trash 0%, Stack Dishware π0 52%, π0.5 48%). 시뮬레이션인지 실제인지는 명시돼 있지 않습니다. 문맥(sim-to-real 실험, "sim-only 미세조정으로 100%")으로 보아 실제 로봇 수치로 보입니다.

### 18.1 결과

| 과제 | 정책 | 이 재현 (25회) | 95% 구간 | 논문 시뮬 | 논문 실제 | 논문 FAQ 6 (공개 체크포인트) |
|---|---|---|---|---|---|---|
| Marker in Cup | π0 | **4/25 = 16%** | 6–35% | 40 | 40 | |
| | π0.5 | **13/25 = 52%** | 33–70% | 88 | 92 | |
| Serve Fruits | π0 | **0/25 = 0%** | 0–13% | 4 | 0 | |
| | π0.5 | **22/25 = 88%** | 70–96% | 80 | 72 | |
| Stack Dishware | π0 | **0/25 = 0%** | 0–13% | 34* | 100* | 52 |
| | π0.5 | **16/25 = 64%** | 45–80% | 64* | 100* | 48 |
| Store Marker | π0 | **0/25 = 0%** | 0–13% | 4* | 48* | 0 |
| | π0.5 | **1/25 = 4%** | 1–20% | 20* | 60* | 0 |
| Throw Away Trash | π0 | **0/25 = 0%** | 0–13% | 0* | 20* | 0 |
| | π0.5 | **0/25 = 0%** | 0–13% | 4* | 48* | 0 |

\* 미세조정한 정책의 값입니다. 여기서 평가한 공개 체크포인트와는 다른 정책입니다.

95% 구간은 Wilson 구간입니다. 논문도 칸마다 25회를 돌렸습니다(부록 J.1).

- **직접 비교할 수 있는 두 과제**:
  - **Serve Fruits는 맞습니다.** π0.5 88%(논문 시뮬 80%), π0 0%(논문 4%)입니다. π0.5는 바나나와 사과를 25회 모두 집어 접시에 올렸습니다(사과 놓기 92%).
  - **Marker in Cup은 두 정책 모두 낮습니다.** π0.5 52%(논문 88%)와 π0 16%(논문 40%)입니다. 25회짜리 두 비율의 차이를 Fisher 정확 검정으로 보면 π0.5는 p = 0.01, π0는 p = 0.11입니다.
  - 두 정책의 순서(π0.5 > π0)는 두 과제 모두 논문과 같습니다.
  - 19장의 GR00T N1.6은 같은 Marker in Cup 장면, 같은 시작 상태에서 5/25(20%)로 논문 28%와 맞습니다. 그러니 이 장면에서 어긋나는 것은 openpi 두 정책입니다.
- **미세조정 행의 세 과제**:
  - 공개 π0.5는 Stack Dishware에서 64%(16/25)입니다. 논문의 미세조정 π0.5 시뮬 값(64%)과 같고, FAQ의 공개 체크포인트 값(48%)보다 높습니다.
  - Store Marker와 Throw Away Trash는 FAQ대로 거의 0입니다. 다만 π0.5가 Store Marker를 한 번 끝까지 해냈습니다. 영상으로 확인하니 서랍 열기, 마커 넣기, 서랍 닫기를 모두 했습니다.
  - Store Marker에서 π0.5는 가운데 서랍을 25회 중 11회 실제로 열지만, 마커를 집은 것은 2회입니다. π0는 서랍을 한 번도 열지 않고 마커부터 집습니다(68%).
  - Throw Away Trash에서는 두 정책 모두 컵은 집지만(π0 64%, π0.5 80%) 뚜껑을 제대로 연 것은 각각 1회입니다.

마일스톤별 비율(25회 중):

| 과제 | 정책 | 마일스톤 |
|---|---|---|
| Marker in Cup | π0 / π0.5 | 마커 집기 88% / 100%, 컵 안 16% / 52% |
| Serve Fruits | π0 / π0.5 | 바나나 집기 12% / 100%, 바나나 놓기 4% / 100%, 사과 집기 12% / 100%, 사과 놓기 12% / 92% |
| Stack Dishware | π0 / π0.5 | 그릇 집기 68% / 88%, 그릇을 접시에 52% / 80%, 컵 집기 16% / 100%, 컵을 그릇에 0% / 76% |
| Store Marker | π0 / π0.5 | 가운데 서랍을 실제로 엶† 0% / 44%, 마커 집기 68% / 8%, 서랍 안 0% / 8%, 서랍 모두 닫힘 0% / 4% |
| Throw Away Trash | π0 / π0.5 | 뚜껑을 실제로 엶† 4% / 4%, 컵 집기 64% / 80%, 통 안 0% / 0%, 닫힘 0% / 0% |

† 관절이 이동 범위의 30% 넘게 움직인 에피소드입니다(`analyze_rollouts.py`가 롤아웃에서 관절 궤적을 읽어 셈). 태스크 설정의 "열림" 마일스톤(범위의 5%)은 두 과제 모두 쓸 수 없었습니다. Store Marker는 서랍이 저절로 나오고(18.4), Throw Away Trash는 뚜껑이 5° 남짓 들썩이는 것까지 셉니다(마일스톤 값 π0 76%, π0.5 24%).

### 18.2 무엇을 어떻게 돌렸나

| 항목 | 이 재현 | 근거 |
|---|---|---|
| 장면 | HF `nadunRanawaka1/simfoundry-assets`의 `scenes/DROID/` 5개. Stack Dishware만 바닥면 높이를 바꿈(18.3) | 저자 공개 |
| 정책 | `gs://openpi-assets/checkpoints/pi0_droid_jointpos`(12.0 GB), `pi05_droid_jointpos`(12.4 GB) | 부록 H: "DROID-pretrained joint-position versions" |
| 정책 서버 | openpi `15a9616`의 `serve_policy.py`, 설정 `pi0_droid_jointpos_polaris` / `pi05_droid_jointpos_polaris` | `install_simfoundry.sh`가 주석 처리한 2.16단계가 고정한 커밋. sim-evals README와 같은 서버 설정. 모델은 현재 관절 기준 변화량을 내고 서버가 절대 관절 위치로 바꿈. 정규화 통계는 체크포인트의 것 |
| 평가 코드 | `scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py`와 `simfoundry/policies/openpi.py` | 저장소. 평가 단계에는 장면에 기록된 질량을 적용하는 호출 하나만 더함(공개 장면에는 질량 기록이 없어 본 평가에는 영향 없음, 18.4) |
| 관측·제어 | 저장소 기본값: 외부 카메라 2대(`nv_franka_droid`, 1280×720)와 손목 카메라 → 224×224, 관절 위치 제어 15 Hz, 물리 120 Hz, 그리퍼 반전·이진화, 장면의 보조 파지(assisted grasping) | 저장소 |
| 로봇 시작 자세 | 매 에피소드 제어기의 리셋 자세 [0, −1.3, 0, −2.87, 0, 2.0, 0] | 저장소(평가 단계가 그대로 둠) |
| 에피소드 | 과제·정책마다 25개. 시작 상태는 물체마다 5×5 격자이고, 두 정책이 같은 25개를 씀(`repro/policy_eval/init_states/`) | 부록 J.1 |
| 지시문 | 부록 I.2의 문장(소문자, 마침표 없음) | 저장소 평가 설정 두 개가 같은 형태 |
| 시간 제한·실행 지평 | 저장소 평가 설정이 있는 과제는 그 값(Marker in Cup 90 s·8, Serve Fruits 120 s·24, Stack Dishware 90 s·8). 없는 두 과제는 120 s·8. 평가 단계는 min(지평, 청크)만큼 실행(π0 청크 10, π0.5 15) | `PutMarkerInCup.yaml`, `ServeFruitsOnGreenPlate.yaml`, `stack_dishware.yaml` |
| 성공 판정 | 과제별 태스크 설정의 목표 조건(`repro/policy_eval/tasks/`). 네 단계 과제는 논문 채점 기준의 마지막 상태(서랍·뚜껑까지 닫힘) | 저장소 설정, 부록 I.2 |

시작 상태 격자(전체 폭, 회전 범위)는 다음과 같습니다.
- Stack Dishware: `scripts/cfg/sample_init_states/example.yaml`이 바로 이 장면용이라 그 값을 그대로 썼습니다.
- Marker in Cup, Serve Fruits: 그 장면의 저장소 태스크 설정에 있는 무작위화 범위를 썼습니다(±r → 2r).
- Store Marker, Throw Away Trash: 설정이 없어서 집는 물체만 20×20 cm로 두었습니다. 마커는 ±30°로 돌리고(예시 설정의 머그 범위), 둥근 컵은 돌리지 않았습니다. 정리함과 쓰레기통은 고정 물체라 움직이지 않았습니다.

### 18.3 저장소에 없어서 채운 것

평가 코드는 그대로 썼지만, 공개 저장소만으로는 이 평가가 돌지 않았습니다.

- **openpi 클라이언트**: `install_simfoundry.sh`의 openpi 단계(2.16)가 "학습 코드와 함께 나중에 공개" 주석과 함께 꺼져 있습니다. 그런데 평가 단계는 `openpi_client`를 무조건 import하므로 시작하자마자 실패합니다. 같은 커밋의 클라이언트를 simfoundry 환경에 설치했습니다.
- **태스크 설정**:
  - Store Marker와 Throw Away Trash는 태스크 설정이 아예 없습니다.
  - Marker in Cup 설정(`task/droid/droid_desk_put_marker_in_cup.yaml`)은 컵을 `white_paper_cup_eomccs_1`로 지정합니다. 그런데 공개 장면에 있는 것은 `cream_mug_bpupsc_1`이라 그대로 쓰면 첫 리셋에서 멈춥니다.
  - 다섯 개를 `repro/policy_eval/tasks/`에 새로 썼습니다. 목표 조건과 마일스톤은 저장소 설정을 따랐고, 설정이 없는 두 과제는 부록 I.2의 채점 기준대로 만들었습니다. Store Marker의 서랍은 논문 사진(그림 4)에서 여는 가운데 넓은 서랍(`drawer_1_link`)입니다.
- **닫힘 판정**: OmniGibson의 `Open` 상태는 자산 메타데이터(`openable_joint_ids`)에서 여는 방향을 읽고, 메타데이터가 없으면 양의 방향을 여는 쪽으로 봅니다. 쓰레기통 뚜껑은 [−101°, 0]에서 0이 닫힌 상태라 `Open`이 거꾸로 읽힙니다(닫혀 있으면 열림). 그래서 닫힌 끝을 태스크 설정에서 받는 선택 상태 `JointClosed`를 `simfoundry/tasks/predicates.py`에 추가했습니다. 기본 임계값은 `Open`과 같은 범위의 5%입니다. 서랍 세 개는 모두 0이 닫힘이라 `Open`과 같은 결과를 냅니다.
- **시작 상태 생성기**: 부록 J.1의 5×5 격자를 만드는 도구는 설정 형식(`scripts/cfg/sample_init_states/`)만 있고 스크립트가 없습니다. 같은 형식을 읽는 `repro/policy_eval/sample_init_states.py`를 썼습니다.
  - 물체마다 25칸을 따로 섞어 짝짓습니다. 앵커 물체는 섞지 않습니다.
  - 물리 30스텝 동안 2 cm 넘게 움직이거나 속도가 5 cm/s를 넘으면, 최대 2 cm 흔들어 다시 시도합니다.
  - 격자끼리 겹쳐 그래도 안 되면 두 에피소드 사이에서 그 물체의 칸을 맞바꿉니다. 그래서 각 물체는 25칸을 정확히 한 번씩 씁니다.
  - 결과: Marker in Cup, Store Marker, Throw Away Trash는 25개 모두 첫 시도에 통과했습니다. Serve Fruits는 물체 6개의 격자가 서로 겹쳐서 14개를 흔들어 다시 시도했고, 칸을 4번 맞바꿨습니다. Stack Dishware는 1개를 흔들고 칸을 2번 맞바꿨습니다.
- **Stack Dishware 바닥 높이**:
  - 이 장면만 droid_v2 배경을 그 배경의 등록 자세 그대로 둡니다. 그래서 그려진 책상 면이 z = 0이고(`droid_v2.background.json`의 `support_surface.canonical_height_m`), 저장된 물체도 z ≈ 0에 놓여 있습니다.
  - 배경 메시에는 책상 면 충돌이 없어서, 보이지 않는 바닥면만이 물체를 받칩니다.
  - 그런데 장면 JSON의 `ground_plane_info`는 −0.02입니다. 배경을 2.5 cm 낮춰 둔 다른 네 장면(droid_v1)의 값입니다.
  - 저자 평가 설정(`task/droid_desk_stack_dishware_eval.yaml`)의 `ground_plane_z_offset: 0.02`는 이름과 달리 바닥 높이를 덮어씁니다(`apply_ground_plane_info`, 첫 공개 버전부터).
  - 그래서 장면 값을 쓰면 물체가 책상 속으로 2 cm 가라앉고, 저자 설정을 쓰면 2 cm 뜹니다. 두 경우를 Isaac Sim에서 확인했습니다. 바닥을 그려진 책상 면인 0.0에 두었습니다.

### 18.4 실패 분석과 시뮬레이션 품질

Marker in Cup이 논문보다 낮은 원인으로 시뮬레이션 쪽 문제를 의심하고 확인했습니다. `repro/policy_eval/analyze_rollouts.py`는 평가가 남긴 롤아웃(스텝마다 직렬화된 시뮬레이터 상태)을 다시 불러와, 에피소드마다 물체의 기울기·들린 높이·최종 위치를 냅니다.

**머그컵이 가볍습니다.**
- 장면의 `cream_mug`는 USD에 질량 80 g이 들어 있습니다. 같은 크기(9.4×7.3×8.8 cm)의 도자기 머그는 300 g 안팎입니다.
- π0.5 기준 실행에서는 실패 12회 중 5회에 컵이 45° 넘게 넘어졌습니다.
- 그래서 질량만 300 g으로 바꾼 장면 사본으로 같은 25개 시작 상태를 다시 돌렸습니다(`repro/policy_eval/ablations/`).
  - 질량은 장면 JSON의 `args.mass`로 기록했고, 평가 단계가 불러온 뒤 적용합니다(`og_utils.apply_scene_masses`, AGENTS.md가 소비자에게 권하는 방식). 이 USD는 80 g 기준의 관성도 지정하고 있어서 같은 비율로 키웠습니다.
  - 물리 뷰에서 0.300 kg과 3.75배의 관성으로 읽히는 것을 확인했습니다(`ablations/check_scene_masses.py`).
- 결과: 컵이 넘어진 에피소드는 0회가 됐지만, 성공은 **14/25(56%)**로 80 g일 때(13/25)와 같습니다(p = 1.0). 실패의 모양만 바뀌어, 컵을 넘어뜨리던 에피소드는 이제 마커를 컵 옆에 떨어뜨리거나 끝까지 들고 있습니다.
- 두 실행을 합치면 같은 25개 시작 상태 중 9개는 둘 다 성공, 7개는 둘 다 실패, 9개는 결과가 갈렸습니다. 정책 샘플링의 무작위성이 커서, 두 실행을 합친 27/50(54%)이 이 설정의 더 나은 추정입니다. 이것도 논문 88%와는 다릅니다(p = 0.004).

**충돌 형상은 입구를 조금만 좁힙니다.** 머그의 충돌 형상은 볼록 조각 32개이고 안이 비어 있습니다. 축을 따라 재 보면, 충돌 형상 안쪽 빈 반경(2.5–3.1 cm)은 그려진 안쪽 벽(2.8–3.3 cm)보다 2–3 mm 작습니다. 단면 약 2.5 cm인 마커에는 여유가 충분합니다.

**실패의 대부분은 정책 쪽입니다.** 영상을 보면 π0.5는 마커를 매번 집어(100%) 컵 바로 위까지 세워 가져갑니다. 실패한 에피소드에서는 컵 옆에 떨어뜨린 뒤 다시 집으러 가지 않고, 남은 시간 동안 빈 책상 위를 맴돕니다.

**카메라 위치와 지시문.** 남은 두 후보도 같은 25개 시작 상태로 시험했습니다.
- 카메라: `nv_franka_droid_no_depth`는 왼쪽 카메라를, 기본 설정이 주석으로만 남긴 "calibrated" 위치(10 cm 바깥)에 둡니다. 깊이 출력도 끄지만 정책은 깊이를 쓰지 않습니다. 결과 **12/25(48%)**.
- 지시문: 저장소 설정(`PutMarkerInCup.yaml`)의 "pick up the white marker, put the marker into the cup". 결과 **14/25(56%)**.

**정리하면, 세 후보 모두 성공률을 바꾸지 못했습니다.** 컵 질량, 카메라 위치, 지시문 문구를 바꾼 실행은 48–56%로, 기준(52%)과 각각 p = 1.0입니다.
- 같은 장면·시작 상태에서 π0.5를 네 번 돌린 셈이라 합치면 53/100(53%, 95% 구간 43–62%)입니다. 논문 88%와 분명히 다릅니다(p = 0.001).
- 원인은 이번에 시험한 범위 밖에 있습니다. 남은 후보는 다음과 같습니다.
  - 렌더 색감: 저자가 튜닝했다는 후처리 설정이 공개되지 않았습니다.
  - 시작 상태 분포: 논문의 격자 크기가 공개되지 않아, 저장소 학습용 무작위화 범위(마커 ±12.5 cm)를 썼습니다.
  - 그 밖에 저자 환경과 다른 점들.
- 같은 장면·같은 시작 상태에서 GR00T N1.6은 논문과 맞습니다(19장, 5/25 대 28%). 장면 자체가 크게 틀리지는 않았다는 쪽의 근거입니다. 다만 openpi 두 정책에서만 생기는 차이가 무엇인지는 아직 모릅니다.

**Store Marker의 서랍이 저절로 나옵니다.**
- 롤아웃의 관절 궤적을 보면, 로봇이 건드리지 않아도 에피소드마다 서랍 세 개가 이동 범위의 8–10%(약 1 cm)까지 미끄러져 나옵니다. π0의 25회는 세 서랍 모두 같은 값(8%, 10%, 9%)에서 끝났고, π0.5에서도 로봇이 건드리지 않은 서랍은 같은 값까지 나왔습니다.
- 원인은 확인하지 않았습니다. 정리함은 약 2.7° 기운 자세로 놓여 있습니다(장면의 자세 쿼터니언). 서랍 관절에는 마찰(0.5)과 감쇠가 지정돼 있지만, 이 정도 이동은 막지 못합니다.
- 그래서 태스크의 "서랍 열기" 마일스톤(범위의 5%)은 아무것도 하지 않아도 달성됩니다. 성공 조건 "서랍 모두 닫힘"은 로봇이 저절로 나온 서랍까지 밀어 넣어야 채워집니다.
- 성공률에는 거의 영향이 없습니다. π0는 마커를 서랍에 넣은 적이 없고, π0.5는 넣은 2회 중 1회가 성공으로 판정됐습니다. 마일스톤 표에는 관절 궤적으로 센 값을 썼습니다.

**그 밖에 본 것.**
- **Serve Fruits의 "green plate"**: 시뮬 자산에서는 청록색 쟁반(`teal_tray`)입니다. 그래도 π0.5는 88% 성공했습니다. π0.5는 성공 조건에 없는 배도 25회 중 20회 들어 옮겼습니다(과일을 모두 차리는 쪽으로 해석). 성공 판정에는 영향이 없습니다.
- **보조 파지와 성공 판정**: Marker in Cup의 성공은 대부분 마커를 쥔 채 컵에 꽂아 넣는 순간에 판정됩니다(영상으로 확인). "로봇이 물체와 접촉하지 않음" 조건이 쥔 채로도 참이 되는 것인데, 장면이 쓰는 보조 파지에서는 쥔 물체와의 손가락 접촉이 잡히지 않는 것으로 보입니다. 저자 설정과 같은 판정이라 그대로 두었습니다.
- **논문 그림의 사진**은 로봇 전체가 보이는 별도 촬영 카메라로 보입니다(사진 안에 기둥에 달린 ZED 카메라가 따로 찍힘). 그래서 정책 입력 카메라의 위치를 맞춰 보는 데는 쓸 수 없었습니다.
- **렌더 설정**: 평가 단계 머리말에는 저자들이 "렌더 후처리 색 보정 파라미터를 튜닝"했다는 메모가 있습니다. 하지만 평가 단계는 렌더 설정을 바꾸지 않고, 공개 장면에도 조명 설정(`lighting_state`)이 없습니다.
- 매 리셋마다 "Failed to reset scene: velocities not minimized"가 나옵니다. 리셋 직후 한 스텝 만에 물체 속도가 1 cm/s를 넘는지 보는 검사라, 물체가 내려앉는 첫 스텝에서 늘 걸립니다. 재시도가 늘어 느려질 뿐 결과에는 영향이 없습니다.

### 18.5 실행 환경과 비용

- **3060에서 하지 않은 이유**: π0.5 가중치만 6.2 GB(bf16)이고 추론에 작업 메모리가 더 듭니다. 평가 프로세스(Isaac Sim, 1280×720 카메라 3대)는 4.7 GB를 씁니다. 합치면 12 GB에 거의 꽉 차는데, 이 컴퓨터의 3060은 다른 세션들이 함께 써서 그만큼을 비워 둘 수 없었습니다.
- **G4**: 처음에는 17.2절 방식으로 Colab G4를 빌려 설치했습니다. JAX 0.5.3이 Blackwell(sm_120)에서 돌았고, π0.5 추론 1회에 58 ms(첫 호출 컴파일 30 s)가 걸렸습니다. 그런데 설치 도중 이 컴퓨터가 재부팅되면서 SSH가 끊겼고, 유휴로 판단된 런타임이 회수됐습니다. 다시 빌리려 하니 계정의 G4 동시 한도(3대)가 다른 세션의 런타임으로 차 있었습니다.
- **L4로 변경**: 사용자 결정으로 L4(24 GB, 12 vCPU, RAM 52 GB, 시간당 1.54 CU)로 바꿨습니다. A100·H100에는 RT 코어가 없어 Isaac Sim이 카메라를 렌더하지 못합니다.
  - L4에서는 GPU 메모리 때문에 정책 서버를 하나씩 띄웁니다(XLA 메모리 비율 0.4). 추론 1회는 π0.5 213 ms, π0 175 ms입니다.
  - 평가 속도는 렌더가 병목입니다(평가 하나로 GPU 사용률 63–93%, CPU는 대부분 놂). 평가 하나만 돌면 초당 5–10스텝이고, 둘을 함께 돌리면 각각 3–5스텝입니다. 시간 제한까지 가는 실패 에피소드 하나가 5–9분 걸립니다.
  - 한 대로는 약 14시간이 걸릴 것으로 보여, 사용자 승인으로 L4 두 대를 더 빌렸습니다. 첫 에피소드부터 마지막 본 평가까지 6.5시간이 걸렸고, 그중 마지막 5시간은 세 대가 나눠 맡았습니다.
- **런타임 준비**: `repro/policy_eval/setup_colab.sh`가 17장의 `setup_g4.sh vulkan`(Vulkan ICD·libGLU)과 같은 버전 고정으로 simfoundry 환경을 만듭니다(L4에서 31분). 이어서 openpi 서버·클라이언트와 두 체크포인트를 받습니다.
  - Colab VM에서는 gcsfs가 VM의 메타데이터 자격 증명을 집어 `gs://openpi-assets`가 401을 냅니다. 그래서 openpi 토크나이저처럼 익명(`token: anon`)으로 받습니다.
  - 런타임이 유휴로 회수되지 않도록 `colab exec`로 커널을 계속 바쁘게 둡니다.
- **비용**: G4 약 0.9시간(7.6 CU)과 L4 세 대 합계 약 18시간(28 CU)으로, 모두 약 36 CU입니다. 그중 약 5 CU는 같은 L4에서 돌린 19장 과제(우리 장면 두 개의 π0·π0.5) 몫입니다. 끝난 뒤 모든 런타임을 `colab stop`으로 종료했습니다.

### 18.6 다시 실행하는 법

```bash
colab new -s sf-g1l4 --gpu L4
# 체크아웃을 /content/SimFoundry에(repro/g4/ 포함), 장면 assets/scenes/DROID와
# assets/backgrounds/mesh_backgrounds/droid_v*를 그 assets/ 아래에 복사한 뒤:
bash repro/policy_eval/setup_colab.sh all
mkdir -p /content/eval/init_states && cp repro/policy_eval/init_states/*.json /content/eval/init_states/
R=repro/policy_eval/run_table_g1.sh       # 시작 상태를 새로 만들려면 $R init
$R servers pi05 && $R all pi05 && $R stop pi05
$R servers pi0 && $R all pi0 && $R stop pi0
$R summary
# 비교 실험(π0.5, Marker in Cup)
python3 repro/policy_eval/ablations/make_mug_mass_scene.py assets/scenes/DROID/droid_desk_put_marker_in_cup 0.30
SCENE_JSON=scenes/DROID/droid_desk_put_marker_in_cup/droid_desk_put_marker_in_cup_scene_state_mug300g.json $R eval pi05 marker_in_cup 25 mug300g
SENSORS_CFG=nv_franka_droid_no_depth $R eval pi05 marker_in_cup 25 calcam
PROMPT="pick up the white marker, put the marker into the cup" $R eval pi05 marker_in_cup 25 prompt_repo
# 실패 분석 (런타임에서, 평가 하나의 results_* 폴더마다)
python repro/policy_eval/analyze_rollouts.py --task marker_in_cup --results <results_*> --out marker_in_cup_pi05.json
```

결과 위치: 요약·분석 JSON과 영상은 이 컴퓨터의 `~/simfoundry_logs/policy_eval/`에 받아 두었습니다(Drive에는 아직 올리지 않음). 과제·정책마다 `eval_results.json`, `rollouts.hdf5`, 에피소드 영상(외부 카메라 2대 + 손목, 180×320)이 있습니다.

### 18.7 수정·추가한 파일

| 파일 | 내용 |
|---|---|
| `simfoundry/tasks/predicates.py`, `pick_place_task.py` | 선택 상태 `JointClosed`(관절이 닫힌 끝에서 범위의 5% 안, 닫힌 끝은 설정에서 받음) |
| `scripts/interactive/light_editor/task_semantics.py` | 위 상태를 단항·불리언 상태 목록에 추가 |
| `tests/test_task_aabb_predicates.py` | `JointClosed` 테스트 3개 |
| `simfoundry/utils/og_utils.py`, `scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py` | 장면에 기록된 질량(`args.mass`)을 불러온 뒤 적용(`apply_scene_masses`). 관성도 같은 비율로 조정. 기록이 없으면 아무것도 하지 않음 |
| `repro/policy_eval/setup_colab.sh` | Colab GPU 런타임 준비 |
| `repro/policy_eval/run_table_g1.sh` | 정책 서버, 시작 상태 생성, 평가, 요약 |
| `repro/policy_eval/table_g1.yaml`, `tasks/*.yaml` | 과제별 설정과 논문 수치, 평가용 태스크 설정 5개 |
| `repro/policy_eval/init_states/*.json` | 이 장에서 쓴 시작 상태(과제당 25개) |
| `repro/policy_eval/sample_init_states.py` | 5×5 격자 시작 상태와 물리 검사 |
| `repro/policy_eval/analyze_rollouts.py` | 롤아웃에서 에피소드별 물체 기울기·들림·최종 위치 |
| `repro/policy_eval/summarize.py`, `ping_policy.py`, `contact_sheet.py` | 요약표(Wilson 구간, Fisher 검정), 서버 예열, 영상 프레임 모음 |
| `repro/policy_eval/ablations/` | 머그 질량 장면 사본 생성과 질량 적용 확인 |

## 19. Table G.1 제로샷 행: 우리가 복원한 장면과 세 번째·네 번째 정책 (2026-10-17~18)

논문의 핵심 정량 결과는 "시뮬레이션 평가가 실제 성능을 예측한다"는 것입니다. Table G.1은 과제마다 여러 정책의 시뮬 성공률과 실제 성공률을 나란히 싣고, 과제별로 두 열 사이의 Pearson r과 MMRV를 냅니다. 초록의 **r 0.911과 MMRV 0.018은 이 7개 과제 값의 평균**입니다. `repro/policy_eval_ours/correlate.py --check-paper`는 Table G.1의 시뮬·실제 열에서 7개 과제의 r과 MMRV를 모두 소수 셋째 자리까지 그대로 다시 계산합니다.

18장은 저자들이 공개한 장면 5개에서 공개 π0·π0.5 체크포인트를 평가합니다. 이 장은 거기에 두 가지를 더합니다.

- **우리가 복원한 장면.** 미세조정 없이 평가하는 과제 4개 중 Cup in Bowl은 저자 장면이 공개되지 않았습니다. 그런데 저장소의 태스크 설정(`scripts/cfg/task/droid/droid_desk_put_cup_in_bowl.yaml`)이 부르는 물체(흰 종이컵, 주황 그릇, 검은 펜)가 예제 영상 PutCupInBowl.mp4의 물체이고, 우리 파이프라인이 그 영상을 복원했습니다(11장 `put_cup_in_bowl_v2`). 그 결과로 장면을 만들어 평가합니다. Marker in Cup도 우리 복원(`put_marker_in_cup_v2`)으로 같은 방식의 장면을 만들어, 18장의 저자 장면 결과와 나란히 봅니다.
- **세 번째 정책.** 정책이 둘뿐이면 과제별 r은 정의상 ±1이라 측정이 되지 않습니다. 논문의 제로샷 열 가운데 공개 DROID 체크포인트가 있는 GR00T N1.6-DROID(`nvidia/GR00T-N1.6-DROID`)를 더했습니다. GR00T N1.7은 DROID 체크포인트를 찾지 못했습니다(기본 모델도 백본 접근 승인이 필요).
- **네 번째 정책.** 남은 공개 체크포인트인 DreamZero-DROID(`GEAR-Dreams/DreamZero-DROID`, 영상과 동작을 함께 내는 14B 월드 액션 모델)를 Colab G4 한 대에서 돌렸습니다(19.4).

### 19.1 장면

`repro/policy_eval_ours/build_droid_scene.py`는 저자 장면 `droid_desk_put_marker_in_cup`을 틀로 씁니다. 로봇과 제어기·센서 설정, 저자들이 맞춘 droid_v1 메시 배경 위치, 바닥면(z = -0.02)은 그대로 두고, 그 장면의 물체 두 개(머그컵, 마커)를 빼고 우리 물체를 넣습니다.

- 물체는 12단계에서 z = 0 위에 안정된 자세 그대로 두고, xy만 옮기고 z는 바닥면 높이만큼 내립니다.
- 집는 물체(Cup in Bowl의 컵, Marker in Cup의 마커)는 저자 장면의 마커 자리에, 넣을 곳(그릇, 컵)은 머그컵 자리에 둡니다. 우리 마커는 저자 마커의 긴 축 방향으로 돌립니다(-2.7°). 펜은 방해물로 그 뒤쪽 (0.78, 0.05)에 둡니다.
- 물체 폴더는 real2sim-assets에서 장면 폴더로 복사하고 `USDObject`로 불러옵니다. 저자 장면과 같은 형식이고, USD 해시가 원래 장면이 기대하는 값과 같은지 확인합니다.
- 질량은 우리 파이프라인의 추정값 그대로입니다: 컵 70 g, 그릇 500 g, 펜 10 g / 마커 37.6 g, 컵 150 g.
- 장면: `assets/scenes/DROID/droid_desk_put_cup_in_bowl_ours/`, `assets/scenes/DROID/droid_desk_put_marker_in_cup_ours/`(물체 포함 46 MB).

`check_scene.py`는 평가 단계와 같은 환경(nv_franka_droid 카메라 1280×720, 바닥면 숨김, 태스크 설정)을 띄워 확인합니다. 두 장면 모두 2초 동안 물체가 움직인 거리 0 mm, 시작 시 마일스톤 0이었습니다. 카메라 화면은 `diagnostics/policy_eval_ours/check_*/`에 있습니다.

### 19.2 시작 상태와 실행

- **시작 상태**: 논문 App. J.1대로 물체마다 5×5 격자 25칸을 씁니다. 18장의 `sample_init_states.py`로 만들었습니다(seed 42). 격자 크기는 저장소 태스크 설정의 범위(±r → 2r)입니다. 집는 물체는 25 cm, 넣을 곳과 펜은 10 cm이고 회전은 없습니다. 두 장면 모두 25칸 전부가 물리 검사를 그대로 통과했습니다.
- **설정**: 지시문은 App. I.2를 소문자로 마침표 없이 씁니다. 시간 제한은 저장소 평가 설정(Cup in Bowl 60 s, Marker in Cup 90 s), 실행 지평은 8입니다(`repro/policy_eval_ours/ours.yaml`, 18장 `table_g1.yaml`과 같은 형식).
- **π0·π0.5**: 18장의 Colab L4 런타임에서 18장 드라이버로 돌립니다(`SPEC=repro/policy_eval_ours/ours.yaml`).
- **DreamZero-DROID**: Colab G4(96 GB) 한 대에서 같은 드라이버로 돌립니다. 서버와 설정은 19.4입니다.
- **GR00T N1.6**: 이 머신의 RTX 3060에서 `repro/policy_eval_ours/run_local_gr00t.sh`로 돌립니다.
  - 서버는 Isaac-GR00T `e29d8fc`(Isaac Lab Arena가 이 체크포인트에 고정한 커밋)를 그 저장소의 uv 환경에서 띄웁니다. 6.8 GB를 쓰고, 추론 1회에 0.24 s가 걸리며, 32스텝의 절대 관절 위치를 냅니다.
  - 에피소드는 저장소의 평가 단계를 그대로 `s15_eval.policy=gr00t`로 씁니다. Isaac Sim까지 합쳐 GPU 11.4 GB입니다.
  - 저자 장면 두 과제(Marker in Cup, Serve Fruits)도 18장 런타임과 같은 시작 상태 파일로 돌립니다.
  - 설치 메모: nvcc가 없어 고정된 flash-attn 2.7.4.post1을 빌드할 수 없었습니다. 그래서 GR00T README가 검증한 2.8.0.post2의 torch 2.7용 wheel을 썼습니다. transformers가 import하는 deepspeed가 `CUDA_HOME`을 요구해서, 추론에 쓰지 않는 deepspeed는 이 환경에서 뺐습니다. 이 회선은 큰 Hugging Face 파일을 받다가 멈추므로 체크포인트는 `curl -C -`로 받아 `/data/models/gr00t/`에 두었습니다(체크섬 확인).

### 19.3 결과

π0·π0.5는 18장 런타임이(가까운 배치는 따로 빌린 L4가), GR00T N1.6은 이 머신이, DreamZero는 Colab G4가 돌렸습니다. 성공 횟수/25(24) 회입니다. DreamZero는 계획한 세 과제만 돌렸습니다(–).

| 과제 (장면) | π0 | π0.5 | GR00T N1.6 | DreamZero | 논문 시뮬 (π0 / π0.5 / GR00T / DreamZero) | 논문 실제 |
|---|---|---|---|---|---|---|
| Cup in Bowl (**우리 복원**) | **20/25 = 80%** | **25/25 = 100%** | **25/25 = 100%** | **4/25 = 16%** | 56 / 92 / 40 / 92 | 88 / 100 / 68 / 100 |
| Cup in Bowl (우리 복원, 가까운 배치) | **20/24 = 83%** | **15/24 = 62%** | **18/24 = 75%** | – | 56 / 92 / 40 / 92 | 88 / 100 / 68 / 100 |
| Marker in Cup (**우리 복원**) | **7/25 = 28%** | **20/25 = 80%** | **5/25 = 20%** | **0/25 = 0%** | 40 / 88 / 28 / 80 | 40 / 92 / 28 / 88 |
| Marker in Cup (저자 장면, 18장) | 4/25 = 16% | 13/25 = 52% | **5/25 = 20%** | – | 40 / 88 / 28 / 80 | 40 / 92 / 28 / 88 |
| Serve Fruits (저자 장면, 18장) | 0/25 = 0% | 22/25 = 88% | **4/25 = 16%** | **4/25 = 16%** | 4 / 80 / 20 / 12 | 0 / 72 / 4 / 8 |

**과제별 시뮬-실제 일치** (`correlate.py`: 우리 시뮬 성공률 대 논문의 실제 성공률. DreamZero를 돌린 과제는 정책 넷, 나머지는 셋입니다. 괄호는 DreamZero를 빼고 정책 셋으로 계산한 값입니다. 논문 값은 정책 다섯 개)

| 과제 (장면) | 정책 | Pearson r | MMRV | 논문 r | 논문 MMRV |
|---|---|---|---|---|---|
| Marker in Cup (**우리 복원**) | 넷 | **0.330** (0.999) | **0.420** (0.000) | 0.995 | 0.008 |
| Marker in Cup (저자 장면) | 셋 | 0.961 | 0.080 | 0.995 | 0.008 |
| Serve Fruits (저자 장면) | 넷 | **0.993** (0.993) | **0.010** (0.000) | 0.960 | 0.016 |
| Cup in Bowl (**우리 복원**) | 넷 | **−0.501** (−0.143) | **0.210** (0.173) | 0.907 | 0.016 |
| Cup in Bowl (우리 복원, 가까운 배치) | 셋 | −0.475 | 0.253 | 0.907 | 0.016 |

- **우리가 영상에서 복원한 Marker in Cup 장면은 정책 셋까지 논문의 핵심 결과를 재현했지만, DreamZero에서 어긋납니다.** 세 정책의 성공률(28 / 80 / 20%)이 논문의 시뮬(40 / 88 / 28%)과 실제(40 / 92 / 28%)를 순서와 크기 모두 따라가, r 0.999에 순위 뒤집힘이 없습니다(MMRV 0). 같은 과제의 저자 공개 장면(18장)보다 논문 값에 가깝습니다. 저자 장면에서는 π0.5가 52%에 그쳤는데, 18장의 분석으로는 머그컵 입구(약 6.5 cm)에 마커를 맞춰 넣지 못하는 실패가 대부분입니다. 우리 장면의 종이컵 입구는 약 8.5 cm입니다. 그런데 DreamZero가 이 장면에서 0%(논문 시뮬 80%, 실제 88%)여서, 정책 넷으로는 r 0.330, MMRV 0.420입니다(19.4).
- **Serve Fruits(저자 장면)는 정책 넷으로도 재현됩니다.** π0 0%, π0.5 88%, GR00T 16%, DreamZero 16%(논문 시뮬 12%)로 r 0.993, MMRV 0.010입니다.
- **Cup in Bowl은 재현되지 않습니다.** π0(80%)와 π0.5(100%)는 논문 실제 값(88, 100%)과 맞습니다. 하지만 GR00T가 시뮬에서 100%로 π0.5와 같은 1위인데, 논문의 실제 값은 세 정책 중 가장 낮은 68%입니다. 이 하나가 r을 −0.143으로 끌어내립니다. 배치를 좁혀도 나아지지 않습니다. 가까운 배치에서는 세 정책이 모두 컵을 거의 매번 집지만(22–24/24), 그릇에 넣는 비율이 달라져 π0 83%, π0.5 62%, GR00T 75%가 됩니다. 실제로 가장 강한 π0.5가 가장 낮아져 r은 −0.475입니다. π0.5는 24회 중 17회 컵을 그릇에 넣었지만, 놓은 뒤 판정까지 마친 것은 15회입니다. 두 배치 모두 우리 장면의 Cup in Bowl은 논문의 순위를 재현하지 못합니다. DreamZero(16%, 논문 실제 100%)를 더하면 r은 −0.501로 더 내려갑니다.
- 정책이 세 개 이상인 과제 셋의 r 평균은 정책 넷으로 0.274입니다(Cup in Bowl을 빼면 0.661). DreamZero를 빼고 정책 셋으로 내면 0.616(Cup in Bowl을 빼면 0.996)입니다. 논문의 같은 세 과제 평균은 0.954, 7개 과제 평균은 0.911입니다. 미세조정 과제 셋(Stack Dishware, Store Marker, Throw Away Trash)은 논문의 미세조정 체크포인트가 공개되지 않아 이 비교에 넣을 수 없습니다(18장).

- GR00T N1.6은 우리 Cup in Bowl 장면에서 25회 모두 성공했습니다. 에피소드 길이의 중앙값은 193스텝(12.9 s), 최대는 537스텝으로 60 s 제한 안에 끝났습니다. 성공 영상을 확인해 보니 컵을 집어 그릇 안에 내려놓고 놓았습니다(`diagnostics/policy_eval_ours/gr00t_cup_in_bowl_success_ep0_ep7.jpg`).
- Marker in Cup에서는 25회 모두 마커를 집었지만(마일스톤 touched·picked 25/25) 컵에 넣은 것은 6회, 놓기까지 마쳐 성공한 것은 5회입니다. 실패한 에피소드는 대부분 마커를 든 채 90 s 제한까지 컵 위를 맴돌거나 책상에 떨어뜨립니다. 20%(95% 구간 9–39%)는 논문 값 28%(시뮬·실제 모두)와 통계적으로 구별되지 않습니다.
- 저자 Marker in Cup 장면에서도 GR00T는 5/25(20%)였습니다(18장 시작 상태 파일). 우리 장면과 저자 장면이 같은 값을 냈고, 둘 다 논문 28%와 구별되지 않습니다. 공개 체크포인트가 논문의 GR00T N1.6처럼 동작한다는 뜻이기도 합니다.
- Serve Fruits(저자 장면)는 4/25(16%)로 논문 시뮬 20%와 구별되지 않습니다. 바나나는 23회 집어 17회 접시에 놓았고, 사과는 14회 집어 9회 놓았습니다. 이 장면은 물체가 많아 두 외부 카메라로는 서버와 합쳐 12 GB를 넘어 렌더러가 메모리 부족으로 죽었습니다. 그래서 외부 카메라를 하나만 둔 저장소 설정(`nv_franka_droid_single_cam`)으로 돌렸습니다(최대 11.5 GB). 정책이 받는 외부 카메라(`external_cam0_left`)는 위치와 해상도가 같고, 빠진 두 번째 카메라는 영상 기록에만 쓰입니다.
- 그러면 Cup in Bowl의 100%(논문 시뮬 40%, 실제 68%)는 체크포인트보다 장면 쪽 원인으로 보입니다. 저장소의 Cup in Bowl 설정은 그릇을 컵 바로 옆(간격 1–5 cm)에, 컵을 로봇 정면 가까이에 두는데, 우리 배치는 둘을 29 cm 이상 떼어 놓았습니다. 그래서 두 격자의 중심을 20 cm로 좁히고 로봇 정면 양쪽에 둔 변형(`cup_in_bowl_ours_near`, 컵 (0.55, -0.10), 그릇 (0.55, 0.10))을 같은 방식으로 평가했습니다.
  - 시작 상태 25개 중 2번은 컵이 그릇 테두리 아래에 들어가 컵 바닥면의 96%가 그릇의 AABB 안에 있었습니다. 로봇이 움직이기 전부터 목표(InsideAABB)가 참이므로 빼고 24회를 돌렸습니다. 다른 5개는 31–38%가 겹치지만 판정 기준 50%보다 작아 두었습니다.
  - 결과는 **18/24(75%)**로, 떨어뜨려 놓은 배치의 25/25와 통계적으로 다릅니다(Fisher p = 0.0096). 실패는 테두리 간격 5 cm 미만에서 조금 더 잦았습니다(12회 중 8회 성공 대 12회 중 10회). 24회 모두 컵을 집었고, 실패는 그릇에 넣기와 놓기에서 났습니다.
  - 배치를 좁히면 논문 실제 값(68%) 쪽으로 내려오지만, 논문 시뮬 값(40%)보다는 여전히 높습니다. 저자 장면이 공개되지 않아 물체 크기와 배치가 얼마나 다른지는 확인할 수 없습니다.

### 19.4 DreamZero-DROID: 네 번째 정책 (2026-10-17~18)

DreamZero-DROID는 DROID 데이터만으로 처음부터 학습한 14B 월드 액션 모델입니다(Wan2.1-I2V-14B 기반). 호출마다 앞으로의 영상과 24스텝의 절대 관절 위치를 함께 냅니다. 세 과제를 다른 정책과 같은 시작 상태·지시문·시간 제한·실행 지평(Marker 8, Cup 8, Serve 24)으로 25회씩 돌렸습니다.

| 과제 (장면) | 성공 | 95% 구간 | 논문 시뮬 / 실제 | Fisher p (대 논문 시뮬) | 마일스톤 (25회 중) |
|---|---|---|---|---|---|
| Marker in Cup (우리 복원) | **0/25 = 0%** | 0–13% | 80 / 88 | 2×10⁻⁹ | 마커 닿음 17, 집음 11, 컵 안 0 |
| Cup in Bowl (우리 복원) | **4/25 = 16%** | 6–35% | 92 / 100 | 7×10⁻⁸ | 컵 닿음 19, 집음 12, 그릇 안 4 |
| Serve Fruits (저자 장면) | **4/25 = 16%** | 6–35% | 12 / 8 | 1.0 | 바나나 집음 25·놓음 25, 사과 집음 5·놓음 4 |

- **Serve Fruits(저자 장면)는 논문과 맞습니다.** 바나나는 25회 모두 쟁반에 올렸지만 사과까지 간 것은 드뭅니다(놓음 4회). 논문이 π0.5의 장점으로 든 지시문 따르기의 차이가 그대로 나옵니다(π0.5는 사과 놓기 92%, 18장). 성공 영상은 `diagnostics/policy_eval_ours/dreamzero/serve_fruits_ep2_success.jpg`입니다.
- **우리 복원 장면 두 과제는 논문보다 훨씬 낮습니다.** Marker in Cup에서는 같은 장면의 π0·π0.5·GR00T가 논문 값과 맞았으므로(19.3), 이 차이는 DreamZero에서만 생깁니다. Cup in Bowl은 GR00T도 논문과 달랐습니다.
  - Marker in Cup은 25회 모두 90 s 제한까지 갔습니다. 우리 장면의 마커는 책상에 세워져 있습니다. DreamZero는 마커에 다가간 뒤, 벌린 손가락 옆이나 사이에 마커를 둔 채 수십 초씩 맴돕니다(`marker_in_cup_ours_ep9_hovers_by_upright_marker.jpg`). 25회 중 7회는 그리퍼를 한 번도 닫지 않았습니다. 서버가 저장한 예측 영상을 보면, 모델도 마커 옆에서 집는 장면을 상상합니다(`marker_in_cup_ours_ep1_predicted_wrist_view.jpg`).
  - 첫 에피소드에서는 팔을 의자 쪽으로 돌려 손목 카메라가 배경 메시의 의자 속으로 들어갔고, 26호출(약 14 s) 동안 손목 영상이 검었습니다(`marker_in_cup_ours_ep0_wrist_black_inside_chair.jpg`). 배경 메시에는 충돌이 없습니다.
  - Cup in Bowl의 성공 4회는 영상으로 확인했습니다. 넘어져 있는 종이컵을 집어 그릇에 넣고 놓았습니다(`cup_in_bowl_ours_ep17_success.jpg`). 실패 21회는 모두 60 s 제한까지 갔습니다.
  - 원인은 가리지 못했습니다(아래 호출 방식, 19.5). 저자 Marker in Cup 장면(논문 80%)으로 몇 회 돌려 장면 쪽 원인인지 가르는 진단은 이번에는 하지 않았습니다.

**호출 방식**
- 서버(`socket_test_optimized_AR.py`)는 호출마다 받은 프레임을 쌓습니다. 첫 호출 뒤로는 마지막 4장을 한 블록으로 씁니다. 월드 모델의 문맥(KV 캐시)은 4호출마다 새로 시작합니다(`max_chunk_size` 4). 시드가 고정이라, 같은 입력을 보내면 다섯 번째 호출이 첫 호출과 같은 출력을 냅니다.
- 우리 클라이언트는 호출마다 현재 프레임 1장을 보냅니다. DreamZero 저장소의 시뮬 평가 클라이언트(`eval_utils/run_sim_eval.py`, 지평 8)와 이 저장소의 원래 클라이언트가 그렇게 합니다. 평가 단계는 청크마다 관측을 하나만 정책에 넘기므로, 평가 코드를 고치지 않고는 다른 방식이 불가능합니다.
- 지평 8(Marker, Cup)에서는 서버가 쓰는 4장이 8스텝 간격(−24, −16, −8, 0)입니다. DreamZero의 테스트 클라이언트(`test_client_AR.py`)가 학습 때 간격으로 보내는 [−23, −16, −8, 0]과 거의 같습니다. 다만 블록 사이가 학습의 24스텝이 아니라 8스텝이라, 이어지는 블록이 16스텝씩 겹칩니다.
- 지평 24(Serve Fruits)에서는 블록 사이가 24스텝으로 학습과 같습니다. 대신 4장의 간격이 24스텝으로 학습의 세 배입니다.
- DreamZero가 논문 값과 맞은 과제는 지평 24인 Serve Fruits(저자 장면)뿐이고, 어긋난 두 과제는 지평 8인 우리 장면입니다. 장면과 지평이 함께 바뀌었으므로, 이 결과만으로는 어느 쪽 때문인지 가를 수 없습니다.
- 그 밖에 점검했고 문제가 없던 것: 카메라 슬롯(로봇 왼쪽 카메라가 1번, 오른쪽이 2번으로 다른 정책과 같음), 그리퍼 상태(0 열림, 1 닫힘으로 DROID와 같음), 동작(첫 동작이 보낸 관절 상태에서 0.03–0.08 rad 안이라 절대 관절 목표가 맞음), 색 순서(예측 영상의 색이 입력과 같음).

**클라이언트** (`simfoundry/policies/dreamzero.py`). 저장소의 원래 클라이언트는 공개 서버와 맞지 않아 DreamZero의 시뮬 평가 클라이언트와 같게 고쳤습니다.
- 외부 카메라 키: 서버는 `exterior_image_0_left`·`exterior_image_1_left`를 받아 모델의 1번·2번 카메라에 넣습니다. 원래 코드는 DROID 이름(1, 2)으로 보내서, 1번 카메라가 2번 자리에 들어가고 2번 카메라는 버려졌습니다.
- 세션: 원래 코드는 세션 ID가 고정이라 앞 에피소드의 문맥이 다음 에피소드로 이어졌습니다. 이제 에피소드마다 새 세션을 씁니다.
- 원래 코드는 청크마다 요청을 두 번 보냈습니다(재시도 루프 뒤에 한 번 더). 상태가 있는 서버에서는 프레임이 두 번 쌓입니다.
- 응답: 서버는 (N, 8) 배열을 그대로 보냅니다. 원래 코드는 `["actions"]`로 읽었습니다.
- 연결 유지: 서버는 추론 중 ping에 답하지 않습니다. DreamZero 클라이언트처럼 60 s 간격, 600 s 한도로 기다립니다.
- `repro/policy_eval/check_dreamzero_client.py`가 이것을 DreamZero의 서버 코드로 확인합니다(모델과 GPU 불필요). 이 머신과 G4 런타임에서 전 항목 통과했습니다.

**서버와 설치** (`repro/policy_eval/setup_dreamzero.sh`)
- `dreamzero0/dreamzero` `ab790c1`을 Python 3.11 conda 환경에 README대로 설치했습니다(`pip install -e .`, cu129 인덱스, torch 2.8.0+cu129에 sm_120 포함). 바꾼 것은 세 가지입니다.
  - `gear` 제외: DreamZero가 쓰지 않는, 이름만 같은 PyPI 패키지입니다(이슈 #95).
  - `deepspeed` 제거: 버전 고정이 없어 0.19.7이 깔렸는데, 고정된 transformers 4.51.3과 서로를 import하다 순환 import로 서버가 시작하자마자 죽었습니다. DreamZero에서 deepspeed를 쓰는 곳은 학습 코드뿐입니다.
  - flash-attn과 Transformer Engine 미설치: 주의 연산이 torch SDPA로 대체됩니다. 길이 정보(패딩 마스크)를 넘기는 호출이 없으면 같은 계산이고, 서버 로그에 `Padding mask is disabled` 경고는 한 번도 없었습니다.
- 체크포인트(46 GB, `tensorrt/` 제외)와 아래 Wan2.1 파일(16 GB)은 G4에서 4분 안에 받았습니다. `config.json`의 `skip_component_loading`을 `true`로 둡니다. 그대로 두면 Wan2.1-I2V-14B의 DiT(약 65 GB)를 따로 받아 RAM에 올린 뒤 체크포인트 가중치로 덮어씁니다. T5·CLIP·VAE 파일(16 GB)은 이 값과 상관없이 읽으므로 함께 받습니다. 체크포인트의 `experiment_cfg/conf.yaml`에는 `eval_bf16: true`가 있어, 모델은 bf16으로 GPU에 올라갑니다.
- GPU가 한 장(`ip_size` 1)이라 CFG의 두 갈래를 차례로 계산합니다. DiT 캐시(`--enable-dit-cache`)는 README대로 켰습니다.
- 서버 로드에 8분(RAM 최대 약 105 GB), 첫 두 호출에 82 s·61 s(torch.compile)가 걸렸습니다. 그 뒤로는 호출당 4.2–5.5 s(중앙값 약 4.5 s)입니다. DreamZero 발표값은 H100 두 장에서 약 3 s입니다. 시뮬레이션까지 합치면 8스텝 청크 하나에 5.1 s, 24스텝 청크에 5.5 s입니다.
- GPU 메모리는 서버 78–88 GB에 Isaac Sim 6 GB로, 97.9 GB 중 최대 93.6 GB였습니다.
- 서버는 세션이 바뀔 때마다 앞 세션의 예측 영상을 저장합니다(79개, 예열 포함).

**비용과 진행**
- G4 한 대를 13.3시간 썼습니다(약 118 CU, 승인한 160 CU 안). 평가가 12.5시간(Marker 6.0, Cup 3.8, Serve 2.8)이고, 나머지는 설치(19분)·서버 로드·예열과 아래 두 문제입니다. 추론은 추정(6–12 s)보다 빨랐지만, 실패한 에피소드가 모두 시간 제한까지 가서 호출이 8,630회로 추정(약 4,600회)의 두 배 가까이 됐습니다.
  - 업로드 중 SSH가 끊긴 뒤, VM에 남은 sshd 세션 때문에 새 연결이 HTTP 429로 막혔습니다(약 15분). 커널을 재시작하고 그 세션을 끝내 풀었습니다.
  - 드라이버가 작업 폴더로 이동한 뒤 상대 경로 `SPEC`을 읽어 정책 종류가 빈 값이 되었고, 첫 시도의 세 과제가 시작하자마자 끝났습니다. `SPEC`을 절대 경로로 바꿔 다시 돌렸습니다.
- 결과: `results/policy_eval_ours/dreamzero/`에 평가 출력(`out/`, 영상·롤아웃 포함), 로그(`logs/`), 서버 예측 영상(`server_videos/`), 요약(`summary_ours.json`, `summary_authors.json`), 정책 넷의 상관(`correlate_four_policies.json`)이 있습니다. 진단 이미지는 `results/diagnostics/policy_eval_ours/dreamzero/`에 있습니다. Drive에도 같은 경로로 올렸습니다.

| 파일 | 내용 |
|---|---|
| `repro/policy_eval/setup_dreamzero.sh` | DreamZero 서버 환경과 체크포인트(위 세 가지 차이, `skip_component_loading`) |
| `repro/policy_eval/run_table_g1.sh` | 정책 `kind: dreamzero`: 서버 시작·예열·중지, 평가 단계에 `s15_eval.policy` 전달, 평가는 한 번에 하나. `SPEC`을 절대 경로로, 스펙 옆 `tasks/`도 복사 |
| `repro/policy_eval/setup_colab.sh` | `client`(openpi 클라이언트만), `miniforge` 단계 |
| `repro/policy_eval/ping_dreamzero.py` | 평가 단계의 클라이언트로 서버 예열과 호출 시간 측정 |
| `repro/policy_eval/check_dreamzero_client.py`, `simfoundry/policies/dreamzero.py` | 공개 서버에 맞춘 클라이언트와 그 점검 |
| `repro/policy_eval/table_g1.yaml`, `repro/policy_eval_ours/ours.yaml` | `dreamzero` 정책 항목과 논문의 DreamZero 수치 |

### 19.5 한계와 다음 단계

- **Cup in Bowl의 배치.** 저자 장면이 공개되지 않아, 물체를 어디에 어떤 범위로 놓았는지는 저장소 설정에서 추정했습니다. 세 정책 모두 배치에 민감했고(π0.5 100→62%, GR00T 100→75%), 어느 배치에서도 논문의 순위가 나오지 않았습니다. 저자 장면의 물체(크기·질량·입구)와 시작 범위를 모르는 한 이 과제는 직접 비교할 수 없습니다. 우리 물체의 질량을 실물에 가깝게 고친 경우는 21장에 있습니다(r 0.45, 여전히 순위가 맞지 않음).
- **정책 수.** 논문의 제로샷 열은 정책 다섯 개이고 여기는 넷입니다. GR00T N1.7은 DROID 체크포인트를 찾지 못했습니다. 정책이 넷이어도 r은 한 정책에 크게 흔들립니다(Marker in Cup, 우리 장면: 0.999 → 0.330).
- **DreamZero의 낮은 성공률.** 우리 장면 두 과제에서 논문보다 크게 낮은 원인을 가리지 못했습니다(19.4). 다음에 돌릴 것은 두 가지입니다. 저자 Marker in Cup 장면에서 DreamZero를 돌려 논문 값이 나오면 우리 장면 쪽이 원인입니다. 우리 Marker 장면에서 실행 지평 24로 돌려 나아지면 호출 방식이 원인입니다.
- **비교 기준.** 실제 성공률은 논문의 값을 그대로 썼습니다. 우리가 실제 로봇으로 잰 것이 아닙니다.
- **Clear Table**은 장면도 영상도 공개되지 않아 하지 않았습니다.
- **GR00T 실행 환경.** 이 머신의 12 GB GPU에서 돌렸습니다. Serve Fruits만 외부 카메라 하나로 돌렸고, 정책이 받는 입력은 같습니다.

```bash
# 장면 만들기 (simfoundry 환경)
python repro/policy_eval_ours/build_droid_scene.py --task cup_in_bowl_ours \
  --asset-dir assets --data-dir Data --dataset-dir deps/BEHAVIOR-1K/datasets/real2sim-assets
python repro/policy_eval_ours/check_scene.py --task cup_in_bowl_ours --out results/diagnostics/policy_eval_ours/check_cup_in_bowl
# GR00T N1.6 (openpi_client 대신 import만 되는 빈 모듈을 스크립트가 만들어 씁니다. 다른 폴더는 OPENPI_STUBS로)
repro/policy_eval_ours/run_local_gr00t.sh server
repro/policy_eval_ours/run_local_gr00t.sh all 25 main repro/policy_eval_ours/ours.yaml:cup_in_bowl_ours ...
# DreamZero-DROID (Colab G4 한 대. 체크아웃을 /content/SimFoundry에, 장면 세 개와 droid_v* 배경을 assets/ 아래에,
# 시작 상태 세 개를 /content/eval/init_states/에 올린 뒤)
bash repro/policy_eval/setup_colab.sh miniforge && bash repro/policy_eval/setup_dreamzero.sh conda
bash repro/policy_eval/setup_colab.sh vulkan &
(bash repro/policy_eval/setup_colab.sh simfoundry && bash repro/policy_eval/setup_colab.sh client) &
bash repro/policy_eval/setup_dreamzero.sh all; wait
/root/miniforge3/envs/simfoundry/bin/python repro/policy_eval/check_dreamzero_client.py --dreamzero /content/dreamzero
R=repro/policy_eval/run_table_g1.sh
SPEC=repro/policy_eval_ours/ours.yaml $R servers dreamzero   # 로드 약 8분, 이어서 예열
SPEC=repro/policy_eval_ours/ours.yaml $R eval dreamzero marker_in_cup_ours 25 main
SPEC=repro/policy_eval_ours/ours.yaml $R eval dreamzero cup_in_bowl_ours 25 main
$R eval dreamzero serve_fruits 25 main
# 집계: 과제별로 우리 시뮬 대 논문 실제의 r·MMRV (π0·π0.5: ~/simfoundry_logs/policy_eval, 가까운 배치: near_colab)
python repro/policy_eval_ours/correlate.py --check-paper --results ~/simfoundry_logs/policy_eval/out:main \
  --results ~/simfoundry_logs/policy_eval/out:ours --results results/policy_eval_ours/near_colab/out:near \
  --results results/policy_eval_ours/gr00t/out:main --results results/policy_eval_ours/dreamzero/out:main
```

## 20. 시뮬레이션 품질 수정: 질량, 렌더, 로봇 위치, 마찰 (2026-10-17)

사용자 요청(메인 재현 세션이 전달)으로 시뮬레이션 자체의 품질 문제 가운데 파이프라인 쪽에서 고칠 수 있는 네 가지를 고쳤습니다(마찰은 20.4). 자동 3DGS 배경의 로봇 시점 문제는 17장(fork 세션)이, B7 태스크 제안은 16장 끝(B7 세션)이 맡습니다.

### 20.1 질량

- **장면에 기록된 질량.** 저장 장면의 `objects_info.init_info[<name>].args.mass`는 OmniGibson이 로드할 때 읽지 않아서, 장면을 쓰는 코드가 로드한 뒤에 적용해야 합니다(AGENTS.md). 평가 단계는 18장에서 `apply_scene_masses`를 쓰게 됐고, 이번에 텔레옵 단계(`2_teleop_og_scene.py`)도 같은 함수를 씁니다. 시뮬레이터를 멈춘 상태에서 적용하고, 이어지는 리셋에서 장면의 자세를 다시 놓습니다.
- **VLM 질량 추정의 부피.** 11단계는 메시가 닫혀 있고 바운딩 박스의 10%를 넘게 채우면 그 부피를, 아니면 박스의 65%를 VLM에 "부피"로 줍니다. 컵·그릇처럼 속이 빈 물체는 거의 후자라서 속까지 찬 부피를 받습니다. Qwen3.5는 거기에 플라스틱 밀도를 그대로 곱해 그릇을 5.51 kg으로 냈습니다(16장).
  - 프롬프트에 그 부피가 무엇인지 적습니다. "겉면이 감싸는 부피로, 속이 빈 내부도 찬 것으로 센 값이다. 용기처럼 속이 비었다면 재료와 질량은 그중 작은 일부다"(`envelope_note`).
  - 닫힌 얇은 껍질 메시(속까지 복원된 컵 등)는 메시 자체의 부피가 재료 부피이므로 그것도 함께 줍니다(`material_volume_cm`, `mass_prompt_volumes`). 우리 장면의 물체는 모두 열린 메시라 이 경우는 없었습니다(20.5: 실제로는 UV 이음매 때문에 열린 것으로 판정된 닫힌 메시였습니다).
  - 인자를 주지 않으면 프롬프트는 원래와 글자 하나까지 같습니다. 16장의 재생 기록과 비교하기 위해서입니다. 11단계와 cousin 단계(B4)가 새 문구를 씁니다.
  - 11단계의 질량 호출을 저장된 입력으로 다시 보낸 결과입니다(`repro/vlm_compare/replay_other_calls.py --only A11`, `MASS_PROMPT=hollow`, 결과 `diagnostics/vlm_compare/mass_prompt/`).

    | 장면 / 물체 | qwen3-vl 원래 → 새 문구 | qwen3.5 원래 → 새 문구 | 실물(대략) |
    |---|---|---|---|
    | PutCupInBowl v2 / 주황 그릇 (부피 5,799 cm³) | 500 → 500 g | **5,510 → 100 g** | 수백 g |
    | PutCupInBowl v2 / 흰 컵 | 70 → 50 g | 20 → 10 g | 약 10 g |
    | PutCupInBowl v2 / 검은 펜 | 10 → 10 g | 12 → 3 g | 약 5–10 g |
    | PutMarkerInCup v1 / 종이컵 | 70 → 50 g | 30 → 25 g | 약 10 g |
    | PutMarkerInCup v1 / 빨간 마커 | 36 → 20 g | 40 → 18 g | 약 15–20 g |
    | PutMarkerInCup v1 / 가위 | 50 → 50 g | 50 → 45 g | 약 50–80 g |
    | PutMarkerInCup v2 / 녹색 무늬 컵 | 150 → 80 g | 25 → 15 g | 약 10 g |
    | PutMarkerInCup v2 / 빨간 마커 | 38 → 15 g | 35 → 18 g | 약 15–20 g |
    | PutMarkerInCup v2 / 가위 | 50 → 50 g | 45 → 35 g | 약 50–80 g |

  - 두 모델 모두 실물 쪽으로 움직였고, Qwen3.5의 그릇 오류(5.51 kg)는 사라졌습니다. 종이컵은 qwen3-vl에서 여전히 실물의 5–8배입니다.
- **재생 스크립트의 GPU 유휴 판정**: `bench_common`이 큐가 빈 ComfyUI를 다른 작업으로 세어, Pipeline B가 ComfyUI를 띄워 둔 동안에는 재생이 영원히 기다렸습니다. B7 세션이 `replay_b7_seeds.py`에만 넣었던 수정을 `bench_common.other_gpu_jobs`로 옮겼습니다.

### 20.2 렌더: 톤매퍼

- OmniGibson도 평가 단계도 렌더 설정을 바꾸지 않아서, 정책이 보는 카메라는 Isaac Sim 기본 톤매퍼 ACES(op 6)로 그려집니다. DROID 배경은 사진으로 텍스처를 입힌 스캔 메시인데, ACES는 그 대비와 채도를 올리고 linear(op 1)는 사진의 색을 그대로 둡니다(13.3의 splat 배경 비교에서 실제 프레임 대비 평균 절대 오차 ACES 0.126, linear 0.032). 저자 평가 스크립트 머리말에는 "렌더 후처리 색 보정 튜닝"이라는 메모가 있지만 그 설정은 공개되지 않았습니다.
- 평가 단계에 `s15_eval.tonemap_op`을 넣었고(기본 null = 예전과 같음), 두 드라이버는 `TONEMAP_OP`으로 받습니다. 설정은 몇 프레임을 렌더한 뒤에야 센서 출력에 반영되므로(3프레임에서는 그대로, 20프레임에서 바뀜) 첫 에피소드 전에 20프레임을 렌더합니다. 화면 비교는 `diagnostics/policy_eval_ours/tonemap/marker_authors_aces_vs_linear.jpg`에 있습니다.
- 같은 시작 상태로 linear와 ACES를 비교했습니다(저자 장면, 25회씩. π0.5는 따로 빌린 L4, GR00T는 이 머신).

  | 과제 / 정책 | ACES (18·19장) | linear | 논문 시뮬 |
  |---|---|---|---|
  | Marker in Cup / π0.5 | 13/25 = 52% | **13/25 = 52%** | 88% |
  | Serve Fruits / π0.5 | 22/25 = 88% | **23/25 = 92%** | 80% |
  | Marker in Cup / GR00T N1.6 | 5/25 = 20% | **3/25 = 12%** | 28% |

- π0도 같은 L4에서 linear로 시작했지만, Marker in Cup 2회와 Serve Fruits 1회(모두 실패)를 마친 뒤 멈췄습니다. π0.5와 GR00T에서 차이가 없어서 L4 시간을 더 쓰지 않았습니다. 결과는 Drive `results/policy_eval_ours/render_colab/`(π0.5)와 `results/policy_eval_ours/gr00t/out/marker_in_cup/s15_eval_gr00t_linear/`(GR00T)에, 로그는 `results/logs/policy_eval_linear/`에 있습니다.
- 톤매퍼는 성공률을 바꾸지 않습니다. 저자 장면 Marker in Cup에서 π0.5가 논문보다 낮은 원인은 렌더 색감이 아닙니다(18장이 질량·카메라·지시문도 배제함). 그래서 평가의 기본값은 그대로 두고, splat 배경처럼 색이 크게 달라지는 경우에 쓸 선택 설정으로 남깁니다.

### 20.3 로봇 위치

- 14단계는 로봇 베이스를 `s14_og.robot_config.position = [-1.0, 0, 0]`에 고정해서, 원점 주변에 복원된 책상 물체에 Franka 팔(약 0.85 m)이 닿지 않습니다(PutMarkerInCup 0.87 m. 12장의 시연은 장면 JSON을 손으로 고쳤음).
- `s14_og.robot_standoff_m`를 넣었습니다. 값을 주면 물체들 중심에서 −x 방향으로 그만큼 떨어진 곳에 베이스를 둡니다(로봇은 +x를 봄). 장면 설정의 기본값은 저자 DROID 장면이 물체를 두는 거리인 0.6 m이고, null이면 예전의 고정 위치입니다. 저장소 원래 코드에도 테이블 기준 0.55 m 앞에 두는 줄이 주석으로 남아 있습니다.
- `put_cup_in_bowl_v2`에서 14단계를 다른 폴더(`s14_og_standoff`)로 다시 돌리니, 베이스가 (−0.46, 0.02, 0)에 놓였고 물체까지 거리는 그릇 0.44 m, 컵 0.67 m, 펜 0.71 m였습니다. 셋 다 팔이 닿는 범위 안입니다.

### 20.4 마찰

11단계가 VLM에게 받은 물체별 마찰 계수는 지금까지 시뮬레이션에 한 번도 들어가지 않았습니다. 원인은 세 가지였습니다.

- **재질을 붙이지 않음.** 11단계는 물체마다 마찰 계수 하나("friction (unitless)")를 받아 `s11_sim/scene_objects_info.json`에 적습니다. 14단계, 텔레옵(`2_teleop_og_scene.py`, `2b_teleop_og_scene_cousins.py`), B6(`6_sample_reconstructed_scene.py`)은 이 값으로 `set_obj_materials`(`simfoundry/utils/og_utils.py`)를 부릅니다. 그런데 이 함수는 재질을 만들기만 하고, 충돌 메시에 붙이는 반복문은 주석 처리돼 있었습니다. 그래서 모든 물체가 PhysX 기본 재질(정지·운동 마찰 0.5, 반발 0)로 돌았습니다.
- **운동 마찰 0.** 반복문만 되살려도 안 됩니다. 재질은 정지 마찰만 정했고, 운동 마찰 줄도 주석이었습니다. `UsdPhysics.MaterialAPI`는 정하지 않은 계수를 0.0으로 읽습니다. 그대로 붙였다면 물체가 미끄러지기 시작하는 순간 그 물체 쪽 마찰은 0이 됩니다.
- **play 중에 붙이면 늦게 들어감.** 14단계와 B6은 시뮬레이터가 도는 중에 이 함수를 불렀습니다. 이렇게 붙인 재질은 USD에는 보이지만 PhysX에는 다음 stop/play 때야 들어갑니다(아래 첫 표). 14단계는 그 뒤에 멈추는 일이 없어서, 가라앉히기와 저장이 모두 기본 재질로 끝났습니다. 텔레옵 두 단계는 `og.sim.stopped()` 안에서 불러서 이 문제는 없었습니다.

**고친 것**
- `set_obj_materials`는 물체의 모든 링크에 있는 충돌 메시에 재질을 붙이고, 운동 마찰을 정지 마찰과 같은 값으로 정합니다.
  - 같은 값으로 둔 이유: VLM이 주는 값이 하나뿐이라 둘로 나눌 근거가 없습니다. 평가 단계(물체 0.5, 손가락 2.0)와 라이트 에디터의 마찰 칸도 두 값을 같게 둡니다. 실제 재료의 운동 마찰은 보통 조금 낮지만, 그 비율을 지어내지는 않았습니다.
  - 반발은 예전처럼 정하지 않습니다(0.0). 지금까지 쓰던 PhysX 기본 재질과 같은 값입니다.
- 14단계와 B6(두 곳)은 이 함수를 물체를 넣는 `og.sim.stopped()` 블록 안에서 부릅니다. 함수는 play 중에 불리면 경고를 남깁니다. 14단계는 붙인 값을 로그에 적습니다(`Friction of iter_1: static 0.400, dynamic 0.400`).
- 평가 단계(`1_eval_policy_og_scene.py`)는 고치지 않았습니다. 모든 물체에 0.5, 그리퍼 손가락에 2.0을 일부러 붙이고, Table G.1 수치가 그 설정에서 나왔습니다.
- PhysX는 접촉한 두 재질의 평균을 씁니다(`physxMaterial:frictionCombineMode` 기본값 average). 바닥(ground plane) 재질은 정지·운동 0.5, 반발 0.8입니다. 그래서 바닥에 놓인 물체에는 (추정값 + 0.5)/2가, 물체끼리는 두 추정값의 평균이 걸립니다.

**확인** (`repro/friction/check_friction.py`, 결과 `diagnostics/friction/`, 로그 `logs/friction/`)
- `put_cup_in_bowl_v2`의 물체 셋을 14단계와 같은 방식으로 올리고, 충돌 shape마다 PhysX가 가진 재질을 직접 읽었습니다(텐서 API `RigidBodyView.get_material_properties`). 표의 값은 정지/운동/반발입니다.

  | 물체 (shape 수) | 11단계 추정 | 붙이기 전 | play 중에 붙임 (예전 14단계·B6) | stopped 안에서 붙임 (지금) | 바닥과의 마찰 |
  |---|---|---|---|---|---|
  | iter_0 주황 그릇 (32) | 0.5 | 0.5 / 0.5 / 0 | 0.5 / 0.5 / 0 | 0.5 / 0.5 / 0 | 0.5 → 0.5 |
  | iter_1 검은 펜 (1) | 0.4 | 0.5 / 0.5 / 0 | **0.5 / 0.5 / 0** (USD는 0.4) | **0.4 / 0.4 / 0** | 0.5 → 0.45 |
  | iter_2 흰 컵 (32) | 0.5 | 0.5 / 0.5 / 0 | 0.5 / 0.5 / 0 | 0.5 / 0.5 / 0 | 0.5 → 0.5 |

  - 이 장면에서는 펜의 추정값만 기본값과 달라서, 실제로 바뀌는 물체는 펜 하나입니다.
- **미끄럼 시험.** 그릇 복사본을 바닥에서 +x 방향 0.5 m/s로 밀었습니다. 미끄러진 거리는 바닥과 합친 운동 마찰 μ로 계산한 v²/(2μg)의 79–99%였습니다. 반복문만 되살린 재질(정지 마찰만 0.5)은 지금 재질보다 2.1배 멀리 미끄러졌습니다.

  | 그릇의 재질 | PhysX 운동 마찰 | 바닥과 평균 μ | 미끄러진 거리 | 예상 |
  |---|---|---|---|---|
  | 붙이지 않음 (예전) | 0.5 | 0.5 | 2.34 cm | 2.55 cm |
  | 정지 마찰만 0.5 (반복문만 되살린 경우) | 0.0 | 0.25 | **5.03 cm** | 5.10 cm |
  | 둘 다 0.5 (지금, 그릇의 추정값) | 0.5 | 0.5 | 2.34 cm | 2.55 cm |
  | 둘 다 0.2 | 0.2 | 0.35 | 3.48 cm | 3.64 cm |
  | 둘 다 1.0 | 1.0 | 0.75 | 1.35 cm | 1.70 cm |

- **14단계 재실행.** 20.3의 `s14_og_standoff`와 같은 설정으로 `Data/put_cup_in_bowl_v2/s14_og_friction`에 다시 돌렸습니다(결과 사본 `diagnostics/friction/s14_og_friction/`).
  - 로그에 세 물체의 마찰이 위 값대로 찍혔습니다.
  - 가라앉은 자세는 펜까지 `s14_og_standoff`와 0.001 mm, 0.001° 단위에서 같습니다. 가라앉는 동안 마찰력이 한계(μN)에 닿지 않으면 μ는 계산에 쓰이지 않습니다. 예전 두 실행(`s14_og`, `s14_og_standoff`)끼리도 같았으니 PhysX는 실행마다 같은 결과를 냅니다.
  - 저장된 세 물체는 속도 0이고 잠든 상태입니다. 검사 스크립트에서도 추정값이 들어간 뒤 100스텝 동안 0.05 mm 넘게 움직인 물체가 없습니다.
- 텔레옵(대화형)과 B6은 돌리지 않았습니다. 두 단계도 같은 함수를 `og.sim.stopped()` 안에서 부르므로 첫 표의 "지금" 열과 같은 경우입니다.

**저장 장면에는 기록하지 않았습니다** (아래 "후속"에서 고침)
- AGENTS.md("The saved-scene contract")는 `objects_info.init_info[<name>].args.link_physics_materials`에 마찰을 적으면 OmniGibson이 로드할 때 붙이므로, 소비자를 고치지 않아도 어디서나 쓰인다고 적고 있습니다. 로드할 때 붙는 것은 맞습니다. 이 인자로 만든 물체는 PhysX에서 0.7/0.7로 읽혔습니다.
- 하지만 OmniGibson 3.8.0은 재질을 붙인 뒤 그 dict의 값을 자기가 만든 `PhysicsMaterial` 객체로 바꿔 넣습니다(`object_base.py`의 `_post_load`). 그 dict는 복사본이 아닙니다. 객체의 init info가 같은 dict를 가리키고(`save_init_info`가 인자를 그대로 기록), 장면을 JSON에서 읽었다면 그 JSON 안의 dict도 같은 것입니다.
- 그래서 이 키가 있는 장면을 로드한 뒤 저장하면 `TypeError: Object of type PhysicsMaterial is not JSON serializable`가 납니다. `og.sim.save()`와 `scene.save()` 모두 확인했습니다. 텔레옵과 평가가 쓰는 DataCollectionWrapper도 시작할 때 `env.scene.save()`를 부릅니다.
- 14단계나 B6 출력에 이 키를 넣으면, 그 장면을 읽는 텔레옵(2, 2b), `scene_json: null`로 돌린 평가, 라이트 에디터의 `settle.py`가 저장하는 순간 멈춥니다. 그래서 넣지 않았습니다. 14단계 JSON은 예전처럼 마찰을 담지 않고, 텔레옵과 B6은 `scene_objects_info.json`에서 마찰을 읽어 붙입니다.
- 라이트 에디터의 마찰 칸이 쓰는 키도 이것이라, 그 칸으로 고친 장면은 같은 문제를 겪습니다. 지금 이 머신의 장면 중 이 키를 가진 것은 없습니다.
- 고치려면 OmniGibson이 받은 dict를 복사해서 써야 합니다. `deps/`는 고칠 수 없으므로 `patches/`의 패치나 업스트림 PR로 해야 하고, 따로 할 일로 남깁니다.

**후속: OmniGibson 패치, 저장 장면에 마찰 기록**
- **패치.** `patches/BEHAVIOR-1K-link-physics-materials.patch`는 `BaseObject.__init__`이 받은 `link_physics_materials`를 링크별로 복사해 두게 합니다(`object_base.py` 한 곳). 만든 `PhysicsMaterial`은 복사본에만 들어가고, init info와 장면 JSON의 dict는 받은 그대로 남습니다. 로봇의 손가락 마찰도 복사본에 들어갑니다.
- **설치.** `deps/`는 직접 고치지 않습니다. `install_simfoundry.sh`가 BEHAVIOR-1K를 고정 커밋에 맞춘 뒤 패치를 붙입니다(`git_safe.sh`의 `git_safe_checkout_patched`, `PATCH_PROVENANCE.md`). 붙은 패치 때문에 작업 트리가 "수정됨"으로 보여서 예전 방식으로는 고정 커밋을 옮기지 못하므로, 옮길 때만 패치를 뗐다가 다시 붙입니다(`tests/test_git_safe_checkout.sh`).
- **이 머신.** 사용자 확인을 받고 `/home/inner/real2sim/SimFoundry/deps/BEHAVIOR-1K`에 붙였습니다. simfoundry env가 쓰는 OmniGibson은 이것 하나라서, 이후 시작하는 모든 작업이 패치된 코드를 씁니다. 업스트림 `main`에도 같은 코드가 `usd_object.py`에 남아 있지만, PR은 올리지 않았습니다(사용자 결정).
- **기록.** `set_obj_materials`는 붙인 마찰을 물체의 init info에도 `{<링크>: {static_friction, dynamic_friction}}`로 적습니다. 링크 이름은 로드한 물체에서 읽습니다(이 장면은 `link_textured_mesh`). 패치가 없는 OmniGibson이면 적지 않고 경고만 남깁니다(생성자 소스에서 패치의 복사를 찾음).
  - 그래서 14단계, B6, 텔레옵이 저장하는 장면에 11단계 추정값이 들어가고, 그 장면을 읽는 쪽은 로드할 때 같은 마찰을 받습니다.
  - 평가 단계는 예전처럼 0.5/2.0을 덮어씁니다.
- **문서.** AGENTS.md와 `docs/INSTRUCTIONS_SCENE_EDITOR.md`의 "소비자를 고치지 않아도 어디서나 쓰인다"를, 다시 저장하려면 이 패치가 있어야 한다고 고쳤습니다. 위에서 `DataCollectionWrapper`라 적은 래퍼의 실제 클래스는 `HDF5CollectionWrapper`입니다.

**확인** (결과 `diagnostics/friction/`, 로그 `logs/friction/`. 값은 정지/운동 마찰)

| 점검 | 패치 전 | 패치 후 |
|---|---|---|
| `check_friction.py` contract: 인자 0.7로 만든 물체의 PhysX 재질 | 0.7 / 0.7 | 0.7 / 0.7 |
| 그 물체 init info의 값, `scene.save()`와 `og.sim.save()` | `PhysicsMaterial`, 둘 다 TypeError | `dict`, 둘 다 ok |
| 14단계 재실행(`s14_og_friction_record`)의 저장 JSON | 마찰 없음(`s14_og_friction`) | 세 물체 모두 `link_physics_materials`: 그릇 0.5, 펜 0.4, 컵 0.5 |
| 그 JSON을 텔레옵처럼 `scene_file`로 로드한 PhysX 재질(`check_saved_scene.py`) | 그릇 0.5, 펜 0.4, 컵 0.5 | 그릇 0.5, 펜 0.4, 컵 0.5 |
| 이어서 `scene.save()`, `og.sim.save()`, `HDF5CollectionWrapper` | 셋 다 TypeError | 셋 다 ok, 다시 저장한 JSON의 마찰도 같음 |
| Franka 손가락 마찰 1.5/1.2, 왼손가락 반발 0.3을 인자로(`check_finger_friction.py`) | PhysX 1.5/1.2(왼쪽 반발 0.3). init info가 `PhysicsMaterial`로 바뀌고 저장 TypeError | PhysX 같음. init info는 준 그대로, 저장 ok |

- 패치 뒤에도 `check_friction.py`의 나머지 항목(붙이는 시점별 재질, 미끄럼 시험, 가라앉힘)은 위 두 표와 같습니다.
- 14단계 재실행의 JSON은 새 키를 빼면 `s14_og_friction`과 같고, 가라앉은 자세도 비트 단위로 같습니다. 이 재실행은 deps에 패치를 붙이기 전에, 같은 커밋의 사본에 같은 패치를 붙여 `PYTHONPATH`로 앞세워 돌렸습니다. 나머지 "패치 후" 값은 패치한 deps에서 얻었습니다.

### 20.5 질량: 열린 메시의 부피

22.1의 바나나 610 g은 11단계가 VLM에 준 부피(304.8 cm³)가 실제의 약 두 배였던 탓도 있습니다. 11단계는 trimesh가 닫히지 않았다고 판정한 메시에 방향 상자 부피의 65%를 줍니다. 휘고 긴 물체는 상자의 대부분이 비어 있습니다.

- **그 메시들은 실제로는 닫혀 있습니다.** 텍스처가 있는 메시는 UV 이음매 위의 꼭짓점을 텍스처 좌표마다 하나씩 따로 둡니다. trimesh는 그 이음매를 열린 모서리로 셉니다.
  - 세 장면(Serve Fruits, Cup in Bowl v2, Marker in Cup v2)의 8단계 메시 14개가 모두 이랬습니다. 열린 모서리의 꼭짓점은 전부, 같은 자리에 UV만 다른 사본이 있는 꼭짓점이었습니다.
  - 꼭짓점을 위치만으로 합치자 12개가 닫힌 부피가 됐습니다. 나머지 2개(사과, 초록 바나나 하나)는 삼각형이 하나씩 빠져 있었습니다. `Data/`의 8단계 메시 38개와 cousin 메시 8개도 모두 같은 방법으로 닫혔습니다.
  - 20.1에서 "우리 장면의 물체는 모두 열린 메시"였던 것도 이 판정 때문입니다. 모든 물체가 상자 65% 규칙을 받았습니다.
- **수정**: `mass_prompt_volumes(..., weld_seams=True)`. 열린 메시를 꼭짓점을 합치고 삼각형 하나짜리 구멍을 메운 뒤(`welded_volume`) 다시 잽니다.
  - 그 메시가 볼록 껍질의 40% 이상을 채우는 꽉 찬 물체면 그 부피를 줍니다. 우리 장면에서 과일·바나나·마커·펜이 여기에 들고, 바나나가 가장 낮은 0.54입니다.
  - 접시·컵·그릇처럼 얇은 물체(0.23 이하)와 가위(0.30–0.40)는 예전처럼 상자의 65%와 20.1의 문구를 받습니다. 진짜 구멍이 남은 메시도 마찬가지입니다.
  - 11단계와 cousin 단계(B4)가 켭니다. 인자를 주지 않으면 예전 부피 그대로입니다. 16장과 20.1의 재생 프롬프트가 글자 하나까지 같은 것을 확인했습니다(물체 17개 × 문구 2개).
  - 불러올 때부터 닫혀 있는 메시는 예전 규칙 그대로입니다(20.1의 얇은 껍질 재료 부피 포함). 8단계 메시는 모두 이음매가 있어서 실제로는 이 경우가 없습니다.
- **얇은 물체는 예전 그대로 둡니다.** 처음에는 합친 메시를 닫힌 메시 규칙에 그대로 태웠습니다. 상자의 10%를 넘으면 그 부피를, 아니면 상자의 65%와 재료 부피를 주는 규칙입니다. 이 변형도 재생했습니다(아래 표의 오른쪽에서 두 번째 열).
  - 종이컵 둘이 55 g, 74 g이 됐습니다(실물 약 10 g). 복원된 벽이 약 3 mm로, 실제 종이컵 벽(0.3 mm 안팎)보다 훨씬 두껍습니다. Qwen3.5는 종이라고 맞게 알아보고도, 받은 재료 부피에 종이 밀도를 그대로 곱했습니다.
  - 그릇은 387 g이 됐습니다(예전 100 g, 실물 수백 g). Qwen3.5 스스로 "보통 150–300 g인데 조금 무겁다"고 적었습니다.
  - 접시 둘은 합친 부피(264·234 cm³)가 상자의 15–17%라 그 부피를 그대로 받았습니다. 그런데 문구가 "속 빈 부분까지 센 부피"라고 하니 다시 작게 깎아 20 g, 35 g을 냈습니다(예전 90 g, 30 g). 사진의 두 접시는 성형 플라스틱으로 보입니다.
  - 그래서 합친 부피는 꽉 찬 물체에만 씁니다. 컵은 21장에서 질량에 따라 실패 양상이 달라진 물체입니다(10 g 컵이 손가락에 밀려남). 그래서 컵을 다시 무겁게 만드는 쪽은 택하지 않았습니다.
- **제안된 다른 두 방법보다 정확합니다.** 11단계가 이미 계산하는 볼록 분해(CoACD) 조각 부피의 합과, 메시 복셀 채우기(가장 긴 변의 1/128 간격)도 재 봤습니다.
  - 과일처럼 둥근 물체에서는 둘 다 합친 부피와 15% 안쪽입니다.
  - 얇은 물체에서는 크게 벗어납니다. 볼록 조각은 휜 벽 사이의 빈 공간까지 감싸고, 복셀은 얇은 벽을 한 칸 두께로 부풀립니다. 접시·컵·그릇에서 CoACD 합은 합친 부피의 1.9–2.6배, 복셀은 1.4–1.7배입니다.
  - 게다가 CoACD는 11단계가 질량을 물은 뒤에 돕니다. 그 부피를 쓰려면 순서도 바꿔야 합니다.

| 물체 (cm³) | 상자 65% (예전) | 합친 메시 | CoACD 합 | 복셀 채우기 |
|---|---|---|---|---|
| 바나나 | 304.8 | 142.6 | 154.6 | 162.8 |
| 사과 | 188.7 | 161.7 | 149.0 | 167.1 |
| 주황 접시 | 1,149.8 | 264.2 | 585.4 | 405.9 |
| 가위 | 233.6 | 51.0 | 81.2 | 69.7 |
| 흰 종이컵 | 597.6 | 79.3 (벽) | 154.0 | 107.6 |
| 주황 그릇 | 5,799.4 | 430.3 (벽) | 1,121.7 | 729.7 |

**질량 재생** (`replay_other_calls.py --only A11`, Qwen3.5, thinking 끔, 저장된 입력). 예전은 `MASS_PROMPT=hollow`(지금 11단계 문구, 예전 부피), 새것은 `MASS_PROMPT=welded`입니다. 예전 쪽은 파이프라인 로그(Serve Fruits 8개)와 20.1의 Qwen3.5 값(나머지 6개)을 하나도 빠짐없이 그대로 다시 냈습니다. 새 쪽도 같은 프롬프트를 받은 다른 재생과 답이 모두 같았습니다.

| 장면 / 물체 | 예전 부피 (cm³) | 새 부피 (cm³) | 예전 질량 | 새 질량 | 얇은 물체도 합친 부피를 쓴 변형 | 실물 / 저자 |
|---|---|---|---|---|---|---|
| Serve Fruits / 배("orange tangerine") | 152.3 | **117.3** | 137 g | **113 g** | 117.3 → 113 g | 저자 240 g |
| Serve Fruits / 바나나 | 304.8 | **142.6** | 610 g | **120 g** | 142.6 → 120 g | 저자 120 g, 실물 약 120–150 g |
| Serve Fruits / 사과 | 188.7 | **161.7** | 123 g | **136 g** | 161.7 → 136 g | 저자 130 g |
| Serve Fruits / 초록 바나나(유령) | 323.8 | **143.8** | 363 g | **140 g** | 143.8 → 140 g | — |
| Serve Fruits / 주황 접시 | 1,149.8 | 1,149.8 | 90 g | 90 g | 264.2 → 20 g | 저자 400 g |
| Serve Fruits / 오렌지 | 165.3 | **145.3** | 162 g | **131 g** | 145.3 → 131 g | 저자 140 g |
| Serve Fruits / 초록 바나나(유령) | 248.2 | **110.3** | 253 g | **105 g** | 110.3 → 105 g | — |
| Serve Fruits / 청록 접시 | 884.0 | 884.0 | 30 g | 30 g | 234.2 → 35 g | 저자 500 g |
| Cup in Bowl v2 / 주황 그릇 | 5,799.4 | 5,799.4 | 100 g | 100 g | 5,799.4(재료 430.3) → 387 g | 수백 g |
| Cup in Bowl v2 / 검은 펜 | 3.3 | **2.6** | 3 g | **2.5 g** | 2.6 → 2.5 g | 약 5–10 g |
| Cup in Bowl v2 / 흰 종이컵 | 597.6 | 597.6 | 10 g | 10 g | 597.6(재료 79.3) → 55 g | 약 10 g |
| Marker in Cup v2 / 가위 | 233.6 | 233.6 | 35 g | 35 g | 51.0 → 40 g | 약 50–80 g |
| Marker in Cup v2 / 녹색 무늬 컵 | 655.5 | 655.5 | 15 g | 15 g | 655.5(재료 86.8) → 74 g | 약 10 g |
| Marker in Cup v2 / 빨간 마커 | 37.6 | **42.0** | 18 g | **15 g** | 42.0 → 15 g | 약 15–20 g |

- **바나나는 610 g에서 120 g이 됐습니다.** 저자 값과 같고, 실물 범위 안입니다. Qwen3.5는 여전히 "점토처럼 보인다"고 적었지만, 부피가 실물 크기가 되자 실제 바나나의 밀도로 계산했습니다. 유령 바나나 둘도 363·253 g에서 140·105 g이 됐습니다.
- **둥근 과일은 부피가 12–23% 줄었습니다.** 오렌지는 162 → 131 g(저자 140 g)입니다. 사과는 밀도를 다르게 골라 오히려 123 → 136 g(저자 130 g)이 됐습니다. 배는 137 → 113 g입니다. 저자 값 240 g과 차이가 큰 것은 우리 배가 더 작게 복원됐기 때문입니다(6.2 × 6.1 × 7.7 cm, 저자 6.3 × 6.4 × 8.8 cm).
- 마커와 펜은 부피 차이가 작아 질량도 거의 같습니다. 접시·그릇·컵·가위는 프롬프트가 예전과 같으므로 질량도 같습니다.
- 남은 차이는 부피가 아니라 VLM의 판단입니다. 접시 둘(90·30 g, 저자 400·500 g)과 가위(35 g, 실물 약 50–80 g)가 그렇습니다.
- 이미 만든 장면의 질량은 바꾸지 않았습니다. Serve Fruits는 22.3의 `serve_fruits_ours_authmass`가 이미 바나나 120 g(저자 값, 새 추정과 같음)으로 평가했습니다. GR00T 성공률은 우리 질량일 때와 차이가 없었습니다(p = 0.78).
- 결과는 Drive `results/diagnostics/vlm_compare/mass_volume/`에 있습니다. `volumes.json`(물체별 부피 후보), `hollow/`, `welded/`, `welded_shell_material/`(채택하지 않은 변형), `summary.json`(재생별 부피와 질량), 재생 로그 `replay_*.log`입니다. `volumes.json`과 `summary.json`은 `repro/vlm_compare/mass_volumes.py`가 씁니다.

```bash
# GPU가 빈 뒤에. 11단계 질량 질문을 저장된 입력으로 다시 보냄
export SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_OLLAMA_MODEL=qwen3.5:9b SIMFOUNDRY_OLLAMA_THINK=0 SIMFOUNDRY_OLLAMA_KEEP_ALIVE=0
for v in hollow welded; do
  MASS_PROMPT=$v python repro/vlm_compare/replay_other_calls.py results/diagnostics/vlm_compare/mass_volume/$v --only A11 \
    --data-root Data --scenes serve_fruits,put_cup_in_bowl_v2,put_marker_in_cup_v2
done
# 부피 후보(합친 메시, CoACD 합, 복셀)와 재생 결과 표. GPU 필요 없음
python repro/vlm_compare/mass_volumes.py results/diagnostics/vlm_compare/mass_volume --data-root Data
```

## 21. 질량을 실물에 가깝게 고친 장면으로 다시 평가 (2026-10-17)

20.1에서 질량 프롬프트를 고쳤지만, 19장의 우리 장면은 고치기 전의 질량으로 평가했습니다. 물체 USD에는 13단계가 예전 실행(qwen3-vl, 원래 문구)의 추정값을 박아 두었습니다. 그래서 실물이 약 10 g인 종이컵이 Cup in Bowl에서는 70 g, Marker in Cup에서는 150 g이었습니다. 이 장에서는 질량만 지금 파이프라인 A(Qwen3.5, 새 문구)가 내는 값으로 바꾸고, 나머지는 그대로 둔 채 세 정책을 다시 평가했습니다.

| 물체 | 19장 (USD 값) | 21장 (장면의 `args.mass`) |
|---|---|---|
| Cup in Bowl / 흰 종이컵 | 70 g | 10 g |
| Cup in Bowl / 주황 그릇 | 500 g | 100 g |
| Cup in Bowl / 검은 펜 | 10 g | 3 g |
| Marker in Cup / 빨간 마커 | 37.6 g | 18 g |
| Marker in Cup / 녹색 무늬 컵 | 150 g | 15 g |

### 21.1 방법

- `build_droid_scene.py`가 물체별 `mass`(kg)를 받습니다. 받은 값은 장면 JSON의 `objects_info.init_info[<name>].args.mass`에 쓰고, 평가 단계가 로드한 뒤 적용합니다(18장의 `apply_scene_masses`). USD는 그대로 둡니다(AGENTS.md 규칙 1).
- `ours.yaml`의 `cup_in_bowl_ours_mass`와 `marker_in_cup_ours_mass`는 19장의 두 과제와 질량만 다릅니다. 장면 JSON끼리 비교해도 차이는 `args.mass` 키뿐입니다. 시작 상태 25개도 19장 파일을 그대로 복사했습니다.
- `check_scene.py`도 평가 단계처럼 장면의 질량을 적용하게 했습니다. 두 장면 모두 질량이 바뀐 것을 확인했습니다(`[mass] white_paper_cup_eqmylj_1: 0.070 -> 0.010 kg`). 2초 동안 움직인 거리는 0 mm였고, 시작할 때 마일스톤은 모두 0이었습니다.
- 마찰은 바꾸지 않았습니다. 평가 단계가 모든 물체에 0.5, 그리퍼 손가락에 2.0을 직접 붙이기 때문입니다(20.4).
- GR00T N1.6은 이 머신에서, π0·π0.5는 새로 빌린 Colab L4에서 돌렸습니다. Colab에서는 19장과 같은 드라이버로 두 과제를 동시에 돌렸고(`TASKS=...`, `all <policy> 2 25 mass`), π0.5 다음에 π0 순서였습니다. 런타임은 설치를 포함해 약 3시간(18장의 L4 단가로 약 5 CU) 썼고, 끝난 뒤 종료했습니다.

### 21.2 결과

| 과제 / 정책 | 19장 질량 | 21장 질량 | 차이 (Fisher p) | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|---|
| Cup in Bowl / π0 | 20/25 = 80% | **18/25 = 72%** | 0.74 | 56% | 88% |
| Cup in Bowl / π0.5 | 25/25 = 100% | **25/25 = 100%** | 1.00 | 92% | 100% |
| Cup in Bowl / GR00T N1.6 | 25/25 = 100% | **21/25 = 84%** | 0.11 | 40% | 68% |
| Marker in Cup / π0 | 7/25 = 28% | **8/25 = 32%** | 1.00 | 40% | 40% |
| Marker in Cup / π0.5 | 20/25 = 80% | **20/25 = 80%** | 1.00 | 88% | 92% |
| Marker in Cup / GR00T N1.6 | 5/25 = 20% | **8/25 = 32%** | 0.52 | 28% | 28% |

**과제별 시뮬-실제 일치** (`correlate.py --draws 20000`. 95% 구간: 각 정책의 성공률을 Beta(k+1, 25−k+1)에서 뽑아 r을 다시 계산한 값)

| 과제 (우리 복원) | 질량 | r | r 95% 구간 | MMRV | 논문 r | 논문 MMRV |
|---|---|---|---|---|---|---|
| Cup in Bowl | 19장 | −0.143 | −0.77..+0.58 | 0.173 | 0.907 | 0.016 |
| Cup in Bowl | 21장 | **0.446** | −0.15..+0.98 | **0.133** | 0.907 | 0.016 |
| Marker in Cup | 19장 | 0.999 | +0.88..+1.00 | 0.000 | 0.995 | 0.008 |
| Marker in Cup | 21장 | **0.984** | +0.77..+1.00 | **0.040** | 0.995 | 0.008 |

- **질량만 바꿨을 때 성공률이 유의하게 달라진 정책은 없습니다.** 가장 크게 바뀐 것은 GR00T의 Cup in Bowl(100→84%, p = 0.11)입니다.
- **새로 생긴 실패는 가벼운 컵에서 나옵니다.** Cup in Bowl에서 컵을 건드리기만 하고 집지 못한 실패가 GR00T 3번, π0 5번이었습니다.
  - 영상(GR00T 0번, π0 0번 에피소드)에서는 손가락이 다가가거나 닫힐 때 10 g 컵이 선 채로 밀려나고, 그리퍼는 빈 채로 닫힙니다. 그 뒤 로봇은 빈 그리퍼로 그릇 위를 맴돕니다.
  - 컵이 쓰러지거나 튕겨 나간 것은 아닙니다. 밀릴 때 버티는 마찰력(μmg)이 70 g일 때의 1/7로 줄었기 때문입니다. 실제 종이컵도 이렇게 밀리므로 물리적으로 그럴듯한 실패입니다.
- **Marker in Cup의 실패 양상은 예전과 같습니다.** 마커를 집은 채 90 s 제한을 넘깁니다(GR00T 17번 모두, π0.5 5번 모두, π0 17번 중 14번).
- **Cup in Bowl의 r은 −0.143에서 0.446으로 올랐지만, 구간이 −0.15..+0.98로 넓습니다.** 핵심 어긋남도 그대로입니다. 실제로는 GR00T(68%)가 π0(88%)보다 낮은데, 우리 시뮬에서는 GR00T(84%)가 π0(72%)보다 높습니다. Marker in Cup은 질량을 바꿔도 논문과 맞습니다(r 0.984).
- **결론: 질량 오차는 Cup in Bowl이 재현되지 않는 원인이 아닙니다.** 남은 후보는 19.4의 장면 배치입니다(저자 장면 미공개). 그래도 고친 질량이 실물에 가까우므로, 이후 우리 장면을 평가할 때는 `_mass` 과제를 쓰는 것이 맞습니다.
- 결과는 Drive `results/policy_eval_ours/mass_colab/`(π0·π0.5)와 `results/policy_eval_ours/gr00t/out/*_mass/`(GR00T)에 있습니다. 판정표는 `diagnostics/policy_eval_ours/mass/correlate_mass.json`에 있고, 19장 성공 횟수는 같은 폴더의 `ch19_counts.json`에서 읽습니다. 로그는 `logs/policy_eval_mass/`와 `results/policy_eval_ours/gr00t/logs/*mass*`에 있습니다.

```bash
python repro/policy_eval_ours/build_droid_scene.py --task cup_in_bowl_ours_mass \
  --asset-dir assets --data-dir Data --dataset-dir deps/BEHAVIOR-1K/datasets/real2sim-assets
python repro/policy_eval_ours/check_scene.py --task cup_in_bowl_ours_mass --out results/diagnostics/policy_eval_ours/check_cup_in_bowl_ours_mass
repro/policy_eval_ours/run_local_gr00t.sh all 25 mass \
  repro/policy_eval_ours/ours.yaml:cup_in_bowl_ours_mass repro/policy_eval_ours/ours.yaml:marker_in_cup_ours_mass
# Colab L4 (setup_colab.sh 뒤): 시작 상태는 <EVAL_ROOT>/init_states/<task>.json
SPEC=repro/policy_eval_ours/ours.yaml TASKS="cup_in_bowl_ours_mass marker_in_cup_ours_mass" \
  repro/policy_eval/run_table_g1.sh all pi05 2 25 mass
python repro/policy_eval_ours/correlate.py --summary results/diagnostics/policy_eval_ours/mass/ch19_counts.json \
  --results results/policy_eval_ours/mass_colab/out:mass --results results/policy_eval_ours/gr00t/out:mass --draws 20000
```

## 22. Serve Fruits를 우리 파이프라인으로 복원해 평가 (2026-10-17)

저장소 예제 영상 `Fruits.mp4`는 Serve Fruits 과제의 책상을 찍은 것입니다. 물체는 저자 장면 `droid_desk_serve_fruits`와 같은 여섯 개(배, 바나나, 사과, 오렌지, 주황 접시, 청록 접시)입니다. 19장의 두 과제처럼 이 영상을 우리 파이프라인으로 복원하고, 그 물체로 Table G.1의 Serve Fruits 행을 다시 평가했습니다.

### 22.1 복원

- Pipeline A를 이 머신에서 돌렸습니다(`Data/serve_fruits`, 1시간 1분). VLM은 Qwen3.5, 이미지 모델은 Qwen-Image-2.1, 질량 프롬프트는 20.1의 새 문구이고, 12 GB용 조정은 v2와 같습니다.
- **5단계가 물체 8개를 떼어 냈습니다.** 실제 물체 6개에 "초록 바나나" 둘이 더해졌습니다.
  - 지우기 단계는 물체를 초록색으로 칠해 표시하고(`removal_outline_color` [0, 255, 0]) 이미지 모델에 지우라고 시킵니다. Qwen-Image-2.1-Turbo는 바나나를 지우지 않고 초록 바나나로 다시 그렸습니다.
  - 그 유령이 다시 물체로 잡혀 두 번 더 떼어졌고, 세 번째 시도 뒤에도 남았습니다(`diagnostics/policy_eval_ours/serve_fruits/s5_removal.jpg`).
  - 평가 장면에서는 뺐습니다. Gemini 이미지 모델에서도 이런지는 확인하지 않았습니다.
  - 이 결함은 24장에서 5단계를 고쳐 막았습니다.
- 배에는 "orange tangerine"이라는 이름이 붙었습니다. 장면에서는 저자 범주를 씁니다.
- **크기**는 저자 물체와 1 cm 안쪽으로 맞습니다. 주황 접시만 23.4×17.7 cm로, 저자의 20.1×21.9 cm와 비율이 다릅니다.
- **질량**(11단계 추정)은 바나나가 610 g입니다(저자 120 g, 실물 약 120–150 g). 두 가지가 겹친 결과입니다.
  - VLM이 무광 표면을 보고 바나나를 점토(밀도 2.0)로 판단했습니다.
  - 11단계의 부피 규칙(닫히지 않은 메시는 방향 상자 부피의 65%)이 휜 바나나에서 304.8 cm³을 내, 실제 부피의 약 두 배였습니다. 20.5에서 고쳤습니다. 메시가 내는 142.6 cm³을 주면 Qwen3.5는 120 g이라고 답합니다.
  - 청록 접시는 30 g으로 나왔습니다(20.1의 속 빈 물체 문구가 작동). 주황 접시는 90 g입니다. 저자(Gemini) 값은 각각 500 g, 400 g입니다.

| 물체 | 우리 크기 (cm) | 저자 크기 (cm) | 우리 질량 | 저자 질량 |
|---|---|---|---|---|
| 바나나 | 17.2 × 11.8 × 3.2 | 19.3 × 12.6 × 4.0 | **610 g** | 120 g |
| 사과 | 6.9 × 6.9 × 6.7 | 7.0 × 6.9 × 6.1 | 123 g | 130 g |
| 청록 접시 | 26.6 × 18.1 × 3.2 | 25.7 × 17.9 × 4.3 | 30 g | 500 g |
| 주황 접시 | 23.6 × 17.9 × 4.6 | 20.1 × 21.9 × 3.1 | 90 g | 400 g |
| 배 | 6.2 × 6.1 × 7.7 | 6.3 × 6.4 × 8.8 | 137 g | 240 g |
| 오렌지 | 6.6 × 6.6 × 6.5 | 6.9 × 6.9 × 6.1 | 162 g | 140 g |

### 22.2 장면과 실행

- `ours.yaml`의 `serve_fruits_ours`는 저자 장면을 틀로 씁니다(로봇, 카메라, 배경, 바닥면). 우리 물체 여섯 개를 저자 물체 자리에 저자 범주로 놓고, 바나나와 두 접시는 저자 물체의 방향으로 돌렸습니다. 이를 위해 `build_droid_scene.py`는 과제마다 다른 틀 장면을 받게 됐습니다.
- 태스크 설정(`tasks/ours_serve_fruits.yaml`)은 저자 Serve Fruits 설정을 이름만 바꾼 것입니다. 우리 물체가 저자 범주를 쓰기 때문입니다.
- 시작 상태: 저자 장면과 같은 격자 설정(seed 42)으로 뽑았고, 25개 중 11개는 겹침 때문에 조금 옮겨서 통과했습니다.
- 확인: 정착하는 동안 움직임 0 mm, 시작할 때 마일스톤 0이었습니다. 청록 접시가 실제 영상처럼 책상의 검은 사각형 위에 놓였습니다. 저자 배치가 실제 배치를 따랐다는 근거입니다.
- GR00T N1.6은 이 머신(외부 카메라 하나, 19장과 같음), π0·π0.5는 Colab L4에서 돌렸습니다. 시간 제한 120 s, 지평 24입니다.

### 22.3 결과

| 정책 | 우리 장면 | 저자 장면 (18·19장) | 차이 (Fisher p) | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|---|
| π0 | **2/25 = 8%** | 0/25 = 0% | 0.49 | 4% | 0% |
| π0.5 | **20/25 = 80%** | 22/25 = 88% | 0.70 | 80% | 72% |
| GR00T N1.6 | **13/25 = 52%** | 4/25 = 16% | **0.016** | 20% | 4% |

| 장면 | r | r 95% 구간 | MMRV | 논문 r | 논문 MMRV |
|---|---|---|---|---|---|
| Serve Fruits (**우리 복원**) | **0.824** | +0.56..+0.97 | **0.000** | 0.960 | 0.016 |
| Serve Fruits (저자 장면, 19장) | 0.993 | | 0.000 | 0.960 | 0.016 |

- **순위는 실제와 같습니다**(π0.5 > GR00T > π0, MMRV 0). π0와 π0.5는 저자 장면과 통계적으로 같은 값을 냈습니다. 우리 물체로 바꿔도 두 정책에는 장면이 같은 과제로 보였다는 뜻입니다.
- **GR00T만 크게 높습니다.** 우리 장면 52%는 저자 장면 16%(p = 0.016)와 논문 실제 4%보다 높아 r을 0.824로 낮춥니다.
- **질량 때문은 아닙니다.** 우리 물체에 저자 물체의 질량(바나나 120 g, 청록 접시 500 g 등)을 넣은 `serve_fruits_ours_authmass`에서도 GR00T는 11/25(44%)였습니다. 우리 질량의 13/25와 차이가 없습니다(p = 0.78).
  - 원인은 방해물인 주황 접시였습니다(22.4).
- 실패 양상: 세 정책 모두 사과를 놓는 단계에서 가장 많이 실패했습니다.
  - GR00T 실패 12번 중 6번과 π0.5 실패 5번 중 4번은, 바나나를 접시에 놓고 사과를 집은 뒤 놓지 못했습니다.
  - π0 실패 23번 중 10번은 바나나만 놓았습니다.
  - GR00T 실패 3번은 마일스톤을 모두 지났지만, 끝났을 때 목표 상태가 아니었습니다.
- 결과는 Drive `results/policy_eval_ours/fruits_colab/`(π0·π0.5)와 `results/policy_eval_ours/gr00t/out/serve_fruits_ours*/`(GR00T)에 있습니다. 판정표는 `diagnostics/policy_eval_ours/serve_fruits/correlate_serve_fruits.json`, 로그는 `logs/policy_eval_fruits/`와 `logs/serve_fruits_A.log`에 있습니다.

```bash
# 복원 (이 머신, local/env.sh를 읽은 셸에서)
SIMFOUNDRY_IMAGE_BACKEND=qwen21 repro/run_local_pipeline.sh A_reconstruction --scene-name serve_fruits \
  --video-fpath "$PWD/docs/assets/example_videos/Fruits.mp4" --env-da3 simfoundry --no-stream \
  --cache-mode --model-cache-dir .cache/simfoundry/model_calls_fruits -- s7_mesh.low_vram=true s5_scene.pda_geometric_backend=depth_pro
python repro/policy_eval_ours/build_droid_scene.py --task serve_fruits_ours \
  --asset-dir assets --data-dir Data --dataset-dir deps/BEHAVIOR-1K/datasets/real2sim-assets
python repro/policy_eval/sample_init_states.py --spec repro/policy_eval_ours/ours.yaml --task serve_fruits_ours \
  --task-yaml repro/policy_eval_ours/tasks/ours_serve_fruits.yaml --out results/policy_eval_ours/gr00t/init_states/serve_fruits_ours.json
EXTERNAL_SENSORS=nv_franka_droid_single_cam repro/policy_eval_ours/run_local_gr00t.sh all 25 main repro/policy_eval_ours/ours.yaml:serve_fruits_ours
SPEC=repro/policy_eval_ours/ours.yaml repro/policy_eval/run_table_g1.sh eval pi05 serve_fruits_ours 25 main   # Colab L4
```

### 22.4 GR00T의 차이는 방해물인 주황 접시에서 나옴

- **시작 상태는 원인이 아닙니다.** 두 장면의 시작 상태는 에피소드마다 같은 격자 칸을 씁니다. 물체 위치 차이는 중앙값 1–2 cm이고, 청록 접시는 같은 자리입니다.
- **갈리는 곳은 바나나를 접시에 놓는 단계입니다.** 에피소드별로 보면, 저자 장면에서 바나나를 집고도 놓지 못한 실패가 9번이었습니다(우리 장면 1번). 저자 장면 영상에서 GR00T는 주황 접시 위를 오가다가 방해물인 배를 집어 옮깁니다(`diagnostics/policy_eval_ours/serve_fruits/gr00t_authors_ep1_orange_plate.jpg`).
- **두 장면의 주황 접시가 다릅니다.** 저자 접시는 낮고 붉은 납작한 접시(20.1 × 21.9 × 3.1 cm)이고, 우리 접시는 테두리가 높고 밝은 오목한 접시(23.6 × 17.9 × 4.6 cm)입니다(`orange_plate_authors_vs_ours_t0.jpg`).
- **우리 장면에서 주황 접시만 저자 것으로 바꿨습니다**(`serve_fruits_ours_authplate`).
  - `build_droid_scene.py`의 `keep`으로 저자 접시를 틀 장면 폴더에서 불러옵니다.
  - 시작 상태는 우리 것에서 접시 항목만 저자의 같은 에피소드 것으로 바꿨습니다.

| GR00T N1.6 | 성공 | 바나나 놓기 | 사과 집기 | 사과 놓기 | 바나나만 놓고 실패 |
|---|---|---|---|---|---|
| 저자 장면 | 4/25 | 17 | 14 | 9 | 9 |
| 우리 장면 | 13/25 | 23 | 22 | 16 | 1 |
| 우리 장면 + 저자 주황 접시 | **4/25** | 21 | 16 | 8 | 8 |

- 접시 하나만 바꿨는데 GR00T가 13/25에서 4/25로 떨어졌습니다(p = 0.016). 저자 장면과 성공 횟수도, 실패 양상도 같습니다.
- **GR00T의 Serve Fruits 성공률은 과제와 상관없는 방해물의 생김새에 좌우됩니다.** 우리 접시는 저자 접시(Gemini를 쓴 파이프라인)보다 깊고 밝아서, GR00T를 덜 헷갈리게 했습니다. 실제 로봇에서 GR00T는 4%이므로, 이 정책의 행동으로는 저자 접시 쪽이 실제와 같은 결과를 냅니다.
- **그렇다고 저자 접시가 생김새로 실물에 더 가까운 것은 아닙니다.** 원본 영상을 보면 주황 접시는 두 장을 겹쳐 놓은 것이고, 테두리가 두 겹으로 보입니다(`diagnostics/policy_eval_ours/serve_fruits/orange_plate_video_stacked.jpg`).
  - 우리 메시의 깊이(4.6 cm, 두 겹)는 이 겹친 접시에 가깝고, 저자 접시(3.1 cm, 한 장)가 더 납작합니다.
  - 우리 접시의 실제 잘못은 가로세로 비율입니다(23.6 × 17.9 cm). 실물은 거의 정사각형이고, 저자 접시는 20.1 × 21.9 cm입니다.
  - 그래서 우리 접시를 저자 접시처럼 얕게 고치지는 않았습니다. 6단계 이미지 모델에 모양을 지키라는 문장을 더해 접시를 다시 그려 보았는데, 결과는 여전히 두 겹의 깊은 접시였습니다. 영상 속 실물이 그렇기 때문입니다.
- 이 결과가 말해 주는 것은, 방해물 하나의 세부(두께, 비율, 색)가 한 정책의 시뮬 성공률을 크게 바꾸고, 그것으로 과제의 r이 0.82와 0.99 사이를 오간다는 사실입니다. 어느 쪽 생김새가 실제 로봇의 GR00T와 같은 반응을 끌어내는지는, 실물과의 기하 비교만으로는 알 수 없었습니다.
- **같은 장면에서 π0·π0.5도 다시 돌렸습니다**(Colab L4). π0.5는 20/25로 우리 접시일 때와 같았고, π0는 5/25였습니다(우리 접시 2/25, p = 0.42). 두 정책은 접시의 영향을 받지 않았습니다.

| 장면 | π0 | π0.5 | GR00T N1.6 | r | r 95% 구간 | MMRV |
|---|---|---|---|---|---|---|
| 저자 장면 | 0/25 | 22/25 | 4/25 | 0.993 | +0.94..+1.00 | 0.000 |
| 우리 복원 | 2/25 | 20/25 | 13/25 | 0.824 | +0.56..+0.97 | 0.000 |
| **우리 복원, 주황 접시만 저자 것** | 5/25 | 20/25 | 4/25 | **0.994** | +0.90..+1.00 | 0.027 |
| 논문 (정책 5개) | | | | 0.960 | | 0.016 |

- **방해물 하나를 바로잡으면 우리가 복원한 Serve Fruits도 논문 값에 맞습니다.** 나머지 다섯 물체는 우리 파이프라인이 만든 그대로입니다. MMRV 0.027은 π0(20%)와 GR00T(16%)의 순서가 실제(0%, 4%)와 반대여서 생긴 값이고, 실제 차이가 4%p라 작습니다.
- π0·π0.5 실행의 원본(에피소드별 기록, 영상)은 남아 있지 않습니다. 결과를 묶는 Colab 호출이 시간 초과로 끊겨 내려받기가 실패했는데, 제 명령이 확인 없이 런타임을 종료했습니다. 성공 횟수는 실행 중에 받아 둔 로그에서 가져왔습니다. π0.5는 `Episode N/25 completed` 25줄 중 성공 20줄, π0는 드라이버의 `Success rate: 5/25` 줄입니다. 출처는 `diagnostics/policy_eval_ours/serve_fruits/authplate_openpi_counts.json`에 적어 두었습니다. GR00T 실행(이 머신)의 원본은 그대로 있습니다.

## 23. 배경 스캔에 찍힌 로봇 (2026-10-17)

사용자가 원래 장면의 로봇과 시뮬레이션의 로봇이 겹치지 않느냐고 물었습니다. 확인해 보니 겹칩니다.

### 23.1 무엇이 겹치나

- 저자 메시 배경 `droid_v1`("DROID Station 1")은 Scaniverse로 찍은 실험실 스캔이고, 그 안에 실제 Franka가 들어 있습니다. 장면은 이 스캔을 로봇 좌표계에 맞춰 두어서, 시뮬 로봇의 베이스가 스캔 로봇의 베이스 위에 놓입니다.
  - 스캔 점을 로봇 기준 좌표로 옮겨 보면, 베이스에서 0.76 m까지 솟았다가 앞으로 꺾이는 팔이 있습니다(`diagnostics/policy_eval_ours/norobot/seg_views.png`의 빨간 점).
  - 그 자세는 배경 파일이 시뮬 로봇에 주는 기본 관절값(`[0, −1.3, 0, −2.87, 0, 2.0, 0]`, 최고 0.58 m)과 다릅니다. 그래서 시작할 때도 완전히 포개지지 않습니다.
- **정책 입력에 들어갑니다.** 시뮬 로봇이 움직이면, 사진 텍스처를 입은 스캔 팔이 정지한 채 외부 카메라 영상에 남습니다(`ghost_frames.jpg`: 시뮬 로봇이 컵을 옮기는 동안 오른쪽 위에 스캔 팔이 그대로 있음). 실제 카메라에는 없는 두 번째 로봇입니다.
- **물리에는 영향이 없습니다.** 배경 메시에는 충돌 설정이 없어 로봇과 물체가 부딪히지 않고, 물체는 따로 둔 바닥면(z = −0.02) 위에 놓입니다.
- 저자 DROID 장면 5개 중 4개(Marker in Cup, Serve Fruits, Store Marker, Throw Away Trash)와 우리 장면 전부가 이 배경을 씁니다. 18–22장의 평가는 모두 이 상태에서 돌았습니다. Stack Dishware가 쓰는 `droid_v2`(Station 4)는 팔 자리에 점이 235개뿐입니다(`droid_v1`은 1,525개).

### 23.2 스캔 로봇 가리기

- `repro/policy_eval_ours/mask_scanned_robot.py`는 같은 USDZ를 참조하는 덧씌움 레이어 `droid_v1_norobot.usd`를 씁니다. 스캔 팔에 해당하는 면을 메시의 `holeIndices`에 넣는 방식이라 원본 스캔은 그대로입니다(AGENTS.md 규칙 1).
  - 팔 찾기: 로봇 베이스 위 상자 안에서, 베이스보다 3 cm 넘게 높은 스캔 꼭짓점들을 2.5 cm 안이면 이어 묶고, 베이스 바로 위 꼭짓점이 든 덩어리를 팔로 봅니다.
  - 스캔 메시는 텍스처 조각마다 끊겨 있어서 메시 연결로는 팔이 이어지지 않았습니다(109개). 기본 관절값으로 계산한 운동학 체인도 위의 자세 차이 때문에 맞지 않았습니다.
  - 결과: 꼭짓점 2,387개, 면 3,552개(전체 135,443개)를 가렸습니다. 로봇 기준으로 x −0.19..0.41, y −0.12..0.13, z 0.03..0.76 m 범위입니다.
- `use_background.py`는 장면 옆에 배경만 바꾼 사본 `<scene>_scene_state_norobot.json`을 씁니다(`_latest`는 그대로).
- **렌더 확인**(`check_scene.py --hide-robot`, `background_compare.jpg`): 시뮬 로봇을 숨기고 저자 Marker in Cup을 렌더했습니다. 원래 배경에서는 두 외부 카메라 모두에 스캔 로봇이 보이고, 가린 레이어에서는 사라지며 그 뒤의 창과 벽이 보입니다. Isaac Sim의 RTX 렌더러가 `holeIndices`를 지킵니다.

### 23.3 정책에 주는 영향

같은 저자 Marker in Cup 장면을 배경만 바꿔 같은 시작 상태 25개로 다시 돌렸습니다. π0·π0.5는 Colab L4에서 18장과 같은 드라이버로(`SCENE_JSON=...norobot.json`), GR00T N1.6은 이 머신에서 돌렸습니다.

| 정책 | `droid_v1` (스캔 로봇 있음, 18·19장) | `droid_v1_norobot` | 차이 (Fisher p) | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|---|
| π0 | 4/25 = 16% | **6/25 = 24%** | 0.73 | 40% | 40% |
| π0.5 | 13/25 = 52% | **10/25 = 40%** | 0.57 | 88% | 92% |
| GR00T N1.6 | 5/25 = 20% | **9/25 = 36%** | 0.35 | 28% | 28% |

| 배경 | r | r 95% 구간 | MMRV |
|---|---|---|---|
| `droid_v1` | 0.961 | +0.51..+1.00 | 0.080 |
| `droid_v1_norobot` | 0.555 | −0.94..+1.00 | 0.080 |

- **스캔 로봇을 가려도 성공률이 유의하게 바뀐 정책은 없습니다.** 실패 양상도 같습니다. 대부분 마커를 집은 채 90 s 안에 컵에 넣지 못합니다(π0.5 15번, GR00T 15번).
- **π0.5가 논문보다 낮은 원인은 유령 팔이 아닙니다.** 가린 배경에서도 40%로, 논문 88%에 한참 못 미칩니다. 18장이 배제한 질량·카메라·지시문, 20.2의 톤매퍼에 이어 후보가 하나 더 빠졌습니다.
- r은 0.961에서 0.555로 내려갔습니다. 하지만 π0.5와 GR00T가 오차 범위 안에서 서로 가까워진 결과이고, 두 구간은 크게 겹칩니다. 순위 뒤집힘(MMRV)은 같습니다. 두 경우 모두 실제로는 π0(40%)가 GR00T(28%)보다 높은데, 시뮬에서는 GR00T가 더 높습니다.
- **기본값은 바꾸지 않습니다.** 가린 레이어가 실제 카메라에 보이는 것과 더 가깝지만, 논문 수치는 저자가 공개한 `droid_v1` 장면에서 나왔을 것이고, 이 A/B에서 차이도 없었습니다. `droid_v1_norobot`은 선택지로 남깁니다.
- 결과는 Drive `results/policy_eval_ours/norobot_colab/`(π0·π0.5)와 `results/policy_eval_ours/gr00t/out/marker_in_cup/s15_eval_gr00t_norobot/`(GR00T)에 있습니다. 판정표는 `diagnostics/policy_eval_ours/norobot/correlate_norobot.json`, 가림 레이어 사본은 같은 폴더의 `droid_v1_norobot.usd`, 로그는 `logs/policy_eval_norobot/`에 있습니다.
- Colab의 첫 π0 실행은 서버의 예열 호출 단계에서 실패해 평가로 넘어가지 못했습니다. 같은 런타임에서 다시 돌려 위 값을 얻었습니다.

```bash
python repro/policy_eval_ours/mask_scanned_robot.py --background assets/backgrounds/mesh_backgrounds/droid_v1.usd \
  --scene assets/scenes/DROID/droid_desk_put_marker_in_cup/droid_desk_put_marker_in_cup_scene_state_latest.json   # usd-core 환경
python repro/policy_eval_ours/use_background.py --tag norobot --background assets/backgrounds/mesh_backgrounds/droid_v1_norobot.usd \
  assets/scenes/DROID/droid_desk_put_marker_in_cup/droid_desk_put_marker_in_cup_scene_state_latest.json
python repro/policy_eval_ours/check_scene.py --spec repro/policy_eval/table_g1.yaml --task marker_in_cup --hide-robot \
  --task-yaml repro/policy_eval/tasks/table_g1_marker_in_cup.yaml \
  --scene scenes/DROID/droid_desk_put_marker_in_cup/droid_desk_put_marker_in_cup_scene_state_norobot.json --out <dir>
SCENE_JSON=scenes/DROID/droid_desk_put_marker_in_cup/droid_desk_put_marker_in_cup_scene_state_norobot.json \
  repro/policy_eval_ours/run_local_gr00t.sh all 25 norobot repro/policy_eval/table_g1.yaml:marker_in_cup
```

## 24. 5단계 유령 물체 수정 (2026-10-17)

22.1에서 5단계가 실제 물체 6개에 "초록 바나나" 둘을 더해 8개를 떼어 냈습니다. 5단계를 고친 뒤 세 장면의 5단계만 새 폴더에 다시 돌려 확인했습니다. 6–14단계는 돌리지 않았습니다.

### 24.1 원인

- mask 방식은 지울 물체를 초록으로 칠해 이미지 모델에 넘깁니다. Qwen-Image-2.1-Turbo는 칠한 바나나 자리에 초록 바나나를 그렸고(iter_1), 진한 초록이라 "초록 잔여" 검사((0, 255, 0) ± 50)를 통과했습니다.
- 다음 VLM 탐지가 그것을 새 물체 "green banana"로 잡았습니다. 5단계는 같은 mask 방식으로 두 번 지우려 했고, 두 번 다 같은 식으로 실패했습니다(iter_3, iter_6).
- 기존 픽셀 검사(`is_valid_removed_obj`: 물체 픽셀 가운데 원본 이미지와 같은 비율)는 두 유령을 잡았습니다(0.4%, 0.1%. 실제 물체는 모두 100%). 하지만 기록만 했고, 기본 설정이 `s6_upsample.include_invalid: true`라 6단계 이후로 그대로 넘어갔습니다.
- Qwen-Image-Edit-2511을 쓴 v1에서도 같은 식의 유리잔 유령 루프가 있었습니다. 그때는 설정에서 mask 방식을 뺐습니다(2장).

### 24.2 수정

- **판정**: 새 탐지의 마스크가 앞서 지운 물체의 마스크(5픽셀 키움) 안에 95% 넘게 들어 있으면, 새 물체가 아니라 그 제거가 남긴 유령으로 봅니다.
  - 기준값은 한 탐지 안에서 중복 마스크를 가르는 `duplicate_mask_intersect_area_threshold`(0.95)를 그대로 쓰되, 한 방향으로만 봅니다. 다시 그린 물체는 표시한 영역 안에 더 작게 그려지기 때문입니다. 이 장면의 유령은 바나나 마스크 안에 99.9% 들어 있지만, 바나나 마스크의 75%만 채웠습니다.
  - 지운 물체가 가리고 있던 실제 물체는 원래 보이던 부분이 그 영역 밖에 있어서 걸리지 않습니다. 이 머신의 5단계 결과 11개에서 각 반복이 지운 마스크에 이 기준을 적용하면, 걸리는 것은 Serve Fruits의 iter_3·iter_6뿐입니다.
- **다시 지우기**: 유령은 다른 물체보다 먼저, 그 물체에 아직 쓰지 않은 방식(mask, outline, bbox 중 남은 것)으로 지웁니다.
  - 이 반복은 이미지만 고치고 물체는 내보내지 않습니다. `skipped_iterations/iter_N.json`(`reason: removal_ghost`)만 쓰고 `obj_cat_list`와 `masked_object`는 쓰지 않으므로, 6단계 이후는 이 반복을 보지 않습니다. 다음 물체의 `base_iter`는 이 반복을 가리킵니다.
- **건너뛰기**: 모든 방식이 실패하면 그 유령은 이후 탐지에서 빼고 `skipped_objects`에 적습니다.
- 반복마다 시도한 방식을 `obj_cat_list`의 새 키 `removal_prompt_types_tried`에 남깁니다. 이어 돌리기 때 같은 상태를 디스크에서 다시 만들기 위해서입니다. 5b 피드백으로 범주나 점을 고정한 반복은 판정하지 않습니다.
- 바꾼 파일: `scripts/pipeline/A_reconstruction/stages/5_decompose_scene.py`, 새 `simfoundry/pipeline/removal_ghosts.py`, 테스트 `tests/test_stage5_removal_ghosts.py`·`tests/test_stage5_resume_guard.py`.
- 한계가 둘 있습니다.
  - 지운 물체 뒤에 95% 넘게 가려져 있던 실제 물체가 드러나면, 유령과 구별되지 않아 빠집니다.
  - 표시 색을 바꿔 다시 지우는 방법은 넣지 않았습니다. 제거 프롬프트가 "bright green"을 글로 적고 있어서, 색을 바꾸려면 프롬프트도 바꿔야 합니다.

### 24.3 확인: Serve Fruits

22장의 응답 캐시 사본으로 5단계만 `Data/serve_fruits_ghostfix`에 다시 돌렸습니다(10분, 1–4단계는 원래 장면을 링크).

- iter_0–2는 캐시가 그대로 맞아, 원래 실행과 같은 물체(배, 노란 바나나, 사과)와 같은 이미지가 나왔습니다.
- iter_3에서 VLM이 다시 "green banana"를 잡았습니다. iteration 1이 지운 영역 안에 100% 들어 있어 유령으로 판정됐고, outline 방식 첫 시도에 지워졌습니다. 물체로는 나가지 않았고, 이후 탐지에서도 다시 잡히지 않았습니다.
- **떼어 낸 물체는 7개입니다.** 배, 바나나, 사과, 주황 접시, 오렌지, 청록 접시, 그리고 검은 사각 틀입니다. 초록 바나나는 없고, 마지막 이미지는 빈 책상입니다.
- 검은 사각 틀은 청록 접시 밑에 깔린 실제 물건입니다(22.2). 유령이 아니라, VLM이 어디까지 물체로 세느냐의 차이입니다.
  - 세 번째 VLM 탐지(iter_6)는 원래 실행에서 유령이 남은 이미지를 보고 [청록 접시, 초록 바나나]를 냈습니다.
  - 이번에는 유령이 없는 이미지를 보고 [청록 접시, 검은 사각 틀]을 냈습니다.
  - 저자 장면에도 없는 물체이므로, 22.2처럼 장면을 만들 때 빼면 됩니다.

| | 원래 (22.1) | 수정 후 |
|---|---|---|
| 떼어 낸 물체 | 8개 (초록 바나나 2개 포함) | 7개 (검은 사각 틀 포함) |
| 초록 바나나 | iter_3·iter_6에서 mask로 지우다 실패, 물체로 나감 | iter_3에서 outline으로 지움, 물체 아님 |
| 마지막 이미지 | 초록 바나나와 사각 틀이 남음 | 빈 책상 |

### 24.4 앞선 두 장면

v2의 두 장면(`put_cup_in_bowl_v2`, `put_marker_in_cup_v2`)의 5단계를 수정한 코드로 다시 돌렸습니다. v2 응답 캐시를 재생만 하는 test mode라서, 원래 실행에 없던 요청이 하나라도 생기면 거기서 멈춥니다.

- 두 장면 모두 끝까지 돌았고, 유령 판정은 한 번도 없었습니다.
- 반복 3개씩, 기록, 마스크, 물체 이미지, 제거 전후 이미지가 원래와 픽셀 단위로 같습니다(`repro/compare_s5_runs.py`). 기록에는 새 키 둘만 더해졌습니다.
- 깊이만 조금 다릅니다(픽셀의 0.4–5.6%, 99퍼센타일 2.2 mm 이하, 최대 5.4 mm). Prior-Depth-Anything이 척도를 맞출 때 고정하지 않은 난수를 더하기 때문입니다. 입력이 같은 iter_0에서도 생기는 실행마다의 차이입니다.

결과는 Drive `diagnostics/s5_ghosts/`에 있습니다(전후 비교 `serve_fruits_ghost_fix.jpg`, 물체 목록, 비교 결과). 장면은 `scenes/serve_fruits_ghostfix.tar`, 응답 캐시는 `model_calls/model_calls_fruits_ghostfix`(22장 캐시에 이번 호출을 더한 것), 로그는 `logs/s5_ghosts/`에 있습니다.

```bash
# 1–4단계는 원래 장면을 링크하고 5단계만 새 폴더에 (local/env.sh를 읽은 셸에서)
d=Data/serve_fruits_ghostfix; mkdir -p $d; for s in s1_video s2_da s3_ground s4_frame; do ln -s ../serve_fruits/$s $d/$s; done
SIMFOUNDRY_IMAGE_BACKEND=qwen21 repro/run_local_pipeline.sh A_reconstruction --scene-name serve_fruits_ghostfix \
  --video-fpath "$PWD/docs/assets/example_videos/Fruits.mp4" --env-da3 simfoundry --no-stream --include 5 \
  --cache-mode --model-cache-dir .cache/simfoundry/model_calls_fruits_ghostfix -- s5_scene.pda_geometric_backend=depth_pro
# v2 장면은 같은 식으로 폴더를 만들고 캐시 재생만 (PutMarkerInCup도 같음)
SIMFOUNDRY_IMAGE_BACKEND=qwen21 SIMFOUNDRY_OLLAMA_MODEL=qwen3-vl:8b-instruct repro/run_local_pipeline.sh A_reconstruction \
  --scene-name put_cup_in_bowl_v2_ghostfix --video-fpath "$PWD/docs/assets/example_videos/PutCupInBowl.mp4" \
  --env-da3 simfoundry --no-stream --include 5 --test-mode --model-cache-dir .cache/simfoundry/model_calls_qwen21 \
  -- s7_mesh.low_vram=true s5_scene.pda_geometric_backend=depth_pro
python repro/compare_s5_runs.py Data/put_cup_in_bowl_v2/s5_scene Data/put_cup_in_bowl_v2_ghostfix/s5_scene
```

## 25. Cup in Bowl을 저장소 태스크 설정의 배치로 (2026-10-18)

19장과 21장의 Cup in Bowl 배치 두 가지는 우리가 추정한 것이었습니다. 저자 배치에 관해 공개된 자료는 저장소의 태스크 설정(`scripts/cfg/task/droid/droid_desk_put_cup_in_bowl.yaml`)뿐입니다. 22.4에서 작은 장면 차이 하나가 GR00T의 성공률을 바꾸는 것을 보았으므로, 이 설정을 그대로 따른 배치로 한 번 평가했습니다. 결과가 어느 쪽으로 나오든 기록하기로 하고 돌렸습니다.

### 25.1 시작 상태

- `repro/policy_eval_ours/sample_task_starts.py`는 태스크 설정으로 환경을 리셋해, 태스크 자신의 무작위 배치 코드(`PickPlaceTask._reset_scene`)로 시작 상태를 뽑습니다.
  - 시드는 42이고, Python·NumPy·torch 난수를 모두 고정합니다.
  - 30스텝 동안 정착시킨 뒤, 2 cm 넘게 움직였거나 마일스톤이 이미 참인 리셋은 버립니다. 25번 리셋이 모두 통과했습니다.
  - 평가 단계와 같게 장면의 질량을 적용합니다.
- 설정의 규칙: 컵은 장면 위치에서 ±12.5 cm 안, 그리고 로봇 기준 작업 영역(x 0.3–0.8 m, y ±0.05 m) 안에 놓입니다. 그릇은 컵의 앞·뒤·좌·우 중 한쪽에 1–5 cm 간격으로 놓입니다. 좌우로 놓을 때는 그릇의 x를 작업 영역 전체에서 뽑습니다.
- 장면 `cup_in_bowl_ours_cfg`는 21장의 질량 수정 장면에서 위치만 바꿨습니다.
  - 컵의 장면 위치를 작업 영역 가운데(로봇 기준 x 0.55, y 0)에 두었습니다. 우리 기존 위치(y −0.19 m)는 설정의 y 범위에서 멀어서, 컵이 경계에만 놓이기 때문입니다.
  - 그릇의 장면 위치는 태스크가 덮어쓰는 자리표시입니다.
  - 펜은 방해물로, 설정이 ±5 cm만 흔드는 물체라 그릇 길에서 비킨 오른쪽 앞(0.45, −0.25)에 두었습니다.
- 뽑힌 시작 상태: 컵은 로봇 기준 x 0.43–0.67 m, y ±5 cm에 놓였습니다. 그릇 방향은 왼쪽 8, 오른쪽 3, 뒤 10, 앞 4입니다. 좌우 배치에서는 그릇이 대각선으로 떨어지기도 해, 중심 간격은 19–43 cm였습니다. 가장 먼 그릇은 베이스에서 0.87 m입니다.

### 25.2 결과

π0·π0.5는 Colab L4, GR00T N1.6은 이 머신에서 돌렸습니다.

| 정책 | 저장소 설정 배치 | 논문 시뮬 | 논문 실제 |
|---|---|---|---|
| π0 | **13/25 = 52%** | 56% | 88% |
| π0.5 | **20/25 = 80%** | 92% | 100% |
| GR00T N1.6 | **18/25 = 72%** | 40% | 68% |

Cup in Bowl을 평가한 네 가지 경우를 모으면 다음과 같습니다.

| 배치 | π0 | π0.5 | GR00T | r (95% 구간) | MMRV |
|---|---|---|---|---|---|
| 떨어뜨린 배치 (19장) | 20/25 | 25/25 | 25/25 | −0.143 (−0.78..+0.57) | 0.173 |
| 가까운 배치 (19장) | 20/24 | 15/24 | 18/24 | −0.475 (−0.99..+0.87) | 0.253 |
| 떨어뜨린 배치, 질량 수정 (21장) | 18/25 | 25/25 | 21/25 | 0.446 (−0.15..+0.98) | 0.133 |
| **저장소 설정 배치, 질량 수정** | 13/25 | 20/25 | 18/25 | **0.137** (−0.80..+0.95) | 0.133 |
| 논문 (정책 5개) | | | | 0.907 | 0.016 |

- **이 배치에서도 재현되지 않습니다.** 실제로는 π0(88%)가 GR00T(68%)보다 높은데, 시뮬에서는 GR00T(72%)가 π0(52%)보다 높습니다.
- π0와 π0.5의 성공률은 이 배치에서 논문의 시뮬 값(56%, 92%)에 가장 가까워졌습니다. GR00T는 논문의 실제 값(68%)에 가깝지만, 논문의 시뮬 값(40%)보다는 높습니다.
- 그릇 방향별로 보면, 그릇이 뒤쪽(먼 곳)에 있을 때 성공이 줄었습니다(π0 4/10, π0.5 8/10, GR00T 6/10). π0는 그릇이 오른쪽에 있을 때 3번 모두 실패했습니다.
- 실패 양상: π0 실패 12번 중 7번은 컵을 건드리지도 못했습니다. GR00T 실패 7번과 π0.5 실패 5번 중 4번은 컵을 집은 뒤 그릇에 넣지 못했고, π0.5의 나머지 1번은 그릇에 넣었지만 놓지 못했습니다.
- **네 경우 모두 순위가 맞지 않았습니다**(r −0.48–0.45). 22.4의 Serve Fruits는 저자 장면의 물체를 하나씩 바꿔 끼워 원인을 찾을 수 있었지만, Cup in Bowl은 저자 장면이 없어 그렇게 할 수 없습니다. 22.4로 보아 방해물(펜)이나 그릇의 복원이 후보이지만, 저자 자산 없이는 확인할 수 없습니다.
- 결과는 Drive `results/policy_eval_ours/cfg_colab/`(π0·π0.5)와 `results/policy_eval_ours/gr00t/out/cup_in_bowl_ours_cfg/`(GR00T)에 있습니다. 시작 상태는 `results/policy_eval_ours/gr00t/init_states/cup_in_bowl_ours_cfg.json`, 판정표는 `diagnostics/policy_eval_ours/cfg_correlate.json`, 로그는 `logs/policy_eval_cfg/`에 있습니다.

```bash
python repro/policy_eval_ours/build_droid_scene.py --task cup_in_bowl_ours_cfg \
  --asset-dir assets --data-dir Data --dataset-dir deps/BEHAVIOR-1K/datasets/real2sim-assets
python repro/policy_eval_ours/sample_task_starts.py --task cup_in_bowl_ours_cfg \
  --out results/policy_eval_ours/gr00t/init_states/cup_in_bowl_ours_cfg.json
repro/policy_eval_ours/run_local_gr00t.sh all 25 main repro/policy_eval_ours/ours.yaml:cup_in_bowl_ours_cfg
SPEC=repro/policy_eval_ours/ours.yaml repro/policy_eval/run_table_g1.sh eval pi05 cup_in_bowl_ours_cfg 25 main   # Colab L4
```

## 26. 팔의 추적 지연: 관절 stiffness를 높여 저자 Marker in Cup 다시 평가 (2026-10-18)

저자 장면 Marker in Cup에서 π0.5는 100회 합계 55%로 논문(88%)보다 크게 낮습니다(18.4). 다른 세션이 롤아웃을 읽어, 시뮬의 팔이 모든 관절에서 명령보다 0.19–0.20 s 늦게 따라간다는 것을 찾았습니다. 원인은 태스크가 매 에피소드 정하는 관절 게인(stiffness 400, damping 80, `simfoundry/tasks/macros.py`)입니다. PhysX의 관절 구동은 목표 속도 0으로 PD 제어를 하므로, 움직이는 목표를 damping/stiffness = 0.2 s 늦게 따라갑니다. 논문 부록 I.1은 "추적 오차를 줄이려고 시뮬의 게인을 더 높게 맞췄다"고 적고 있습니다. 그래서 지연을 줄인 경우를 시험했습니다.

### 26.1 방법

- `PickPlaceTask`에 선택 인자 `arm_kp_scale`을 넣었습니다. 값을 주면 팔 관절 7개의 stiffness에만 곱하고 damping은 그대로 둡니다. 기본값 None은 예전과 같습니다.
- `marker_in_cup_kp4`는 저자 Marker in Cup 장면과 시작 상태 25개를 그대로 쓰고, stiffness만 4배(1600)로 합니다. 예상 지연은 0.05 s이고, 관절 관성을 0.5 kg·m²로 잡으면 감쇠비는 약 1.4로 여전히 과감쇠입니다.
- `check_scene.py`가 리셋 뒤 팔 게인을 출력합니다. `arm_kp_scale: 4`에서 stiffness 1600, damping 80이었고, 기본값에서는 400, 80이었습니다.
- 롤아웃으로 잰 추적(다른 세션의 `arm_tracking.py`, GR00T 실행):

  | | 기본 (400/80) | stiffness ×4 (1600/80) |
  |---|---|---|
  | 관절 지연 τ | 0.19–0.20 s | **0.039–0.047 s** |
  | 한 동작 안에 줄이는 간격 | 25–27% | 58–65% |
  | 손끝 오차 중앙값 / p95 | 8.6 / 34.1 mm | **4.5 / 18.3 mm** |

### 26.2 결과

같은 시작 상태 25개로 π0·π0.5는 Colab L4, GR00T N1.6은 이 머신에서 돌렸습니다. 기준값은 18·19장(π0·GR00T 25회)과 18.4의 100회 합계(π0.5)입니다.

| 정책 | 기준 (400/80) | stiffness ×4 | 차이 (Fisher p) | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|---|
| π0 | 4/25 = 16% | **3/25 = 12%** | 1.00 | 40% | 40% |
| π0.5 | 55/100 = 55% (첫 25회 13/25) | **17/25 = 68%** | 0.27 | 88% | 92% |
| GR00T N1.6 | 5/25 = 20% | **4/25 = 16%** | 1.00 | 28% | 28% |

| | r | r 95% 구간 | MMRV |
|---|---|---|---|
| 기준 | 0.961 | +0.51..+1.00 | 0.080 |
| stiffness ×4 | 0.971 | +0.80..+1.00 | 0.080 |

- **팔이 명령을 0.2 s에서 0.04 s 만에 따라가게 해도, 성공률이 유의하게 바뀐 정책은 없습니다.** π0.5의 실행에서도 지연은 0.038–0.047 s, 손끝 오차 중앙값은 3.3 mm로 줄었습니다.
- π0.5는 68%로 기준 55%보다 13%p 높지만, 25회로는 기준과도(p = 0.27), 논문 88%와도(p = 0.17) 구별되지 않습니다. 같은 시작 상태로 75회를 더 돌려 100회로 판정했습니다(26.3).
- π0와 GR00T는 지연을 줄여도 그대로였습니다. 두 정책의 차이(π0 12%, GR00T 16%)는 실제(40%, 28%)와 순서가 반대인 채로 남아, MMRV도 0.08로 같습니다.
- 결과는 Drive `results/policy_eval_ours/kp4_colab/`(π0·π0.5)와 `results/policy_eval_ours/gr00t/out/marker_in_cup_kp4/`(GR00T)에 있습니다. 판정표는 `diagnostics/policy_eval_ours/kp4_correlate.json`에 있습니다.

### 26.3 π0.5 100회 재현

18.4의 재현과 같은 방식으로, 같은 시작 상태 25개를 세 번 더 돌렸습니다(Colab L4, 서버 하나에 평가 둘을 동시에, 그다음 하나).

| π0.5, 저자 Marker in Cup | 각 25회 실행 | 합계 | 95% 구간 (Wilson) |
|---|---|---|---|
| 기준 (400/80, 18.4) | 13, 12, 14, 16 | 55/100 = 55% | 45–64% |
| stiffness ×4 (1600/80) | 17, 11, 13, 13 | **54/100 = 54%** | 44–63% |

- **팔의 추적 지연은 원인이 아닙니다.** 두 설정은 같습니다(p = 1.00). 차이의 95% 구간은 −15..+13%p로, 논문과의 차이 33%p를 설명하지 못합니다. 첫 실행의 68%는 네 번 중 가장 높은 값이었고, 18.4의 좁은 격자 + 지평 15 실행처럼 25회짜리 실행의 흔들림이었습니다.
- 지연을 줄인 팔(0.04 s)로도 논문의 88%와는 여전히 다릅니다(p = 0.002).
- 그래서 저자 장면에서 π0.5가 낮은 원인 후보에서 다음이 모두 빠졌습니다: 질량·카메라·지시문(18장), 시작 격자와 실행 지평(18.4), 톤매퍼(20.2), 배경 스캔의 로봇(23장), 팔의 추적 지연(26장).
- 원인은 아직 모릅니다. 우리 복원 장면(19·21장)에서는 π0.5가 80%로 논문에 가까우므로, 저자 장면의 물체(머그컵의 입구 6.5 cm, 18장)와 시뮬 물리의 조합이 남은 후보입니다.
  - 충돌 메시가 입구를 좁히는지 확인했습니다. 볼록 조각 64개를 위에서 투영해 중심축에서 가장 가까운 조각까지 재면, 비어 있는 입구는 저자 머그가 약 4.8 cm(겉모습 6.4 cm), 우리 종이컵이 약 5.4 cm(겉모습 8 cm)입니다. 투영이라 실제보다 좁게 잰 값입니다. 두께 2 cm 남짓한 마커에는 둘 다 충분하고 차이도 작아서, 이것만으로 설명되지는 않습니다.
- 결과는 Drive `results/policy_eval_ours/kp4_colab/out/marker_in_cup_kp4/`의 `s15_eval_pi05_main`, `rep2`–`rep4`에 있고, 로그는 `logs/policy_eval_kp4rep/`에 있습니다.

### 26.4 physical 잡기

- 평가 단계(`1_eval_policy_og_scene.py`) 머리말에는 저자의 시뮬 튜닝 메모가 있습니다(NOTES FROM JOSIAH).
  - 항목: 렌더 후처리 색 보정, 실제 데이터의 동작을 재생해 궤적 맞추기, 3DGS 배경을 메시 배경으로 바꾸기, **assisted 잡기를 physical 잡기로 바꾸기**, 모든 물체에 마찰 추가, 물체·로봇 자세와 크기 맞추기, 손가락 최고 속도 낮추기.
  - 공개된 장면의 로봇은 `grasping_mode: assisted`입니다. Serve Fruits 설정의 평가 블록에는 `grasping_mode: assisted # physical # assisted`가 남아 있습니다. 우리 평가는 모두 assisted였습니다.
  - assisted에서는 그리퍼가 물체를 잡는 순간 둘이 고정 관절로 묶입니다. 그래서 마커가 머그 테두리에 닿아도, 실제처럼 그리퍼 안에서 조금 돌거나 미끄러져 맞춰 들어갈 수 없습니다.
- 두 드라이버에 `GRASPING_MODE`를 넣어(`++s15_eval.grasping_mode`), 저자 Marker in Cup을 같은 시작 상태로 physical 잡기에서 돌렸습니다. 장면 기록에 `grasping_mode: physical`이 들어간 것도 확인했습니다.

| 정책 | assisted (18·19장) | physical | 차이 (Fisher p) |
|---|---|---|---|
| π0 | 4/25 = 16% | **2/25 = 8%** | 0.67 |
| π0.5 | 55/100 = 55% | **11/25 = 44%** | 0.37 |
| GR00T N1.6 | 5/25 = 20% | **5/25 = 20%** | 1.00 |

- **잡기 방식도 원인이 아닙니다.** π0.5는 오르지 않았고, 논문 88%와도 여전히 다릅니다(p = 0.002). π0.5의 실패 14번은 모두 마커를 집은 채 컵에 넣지 못한 경우로, assisted 때와 같습니다.
- 세 정책의 physical 값으로 낸 r은 0.872, MMRV는 0.080입니다(assisted 기준 실행은 0.961, 0.080).
- GR00T는 성공 횟수와 실패 집계가 우연히 같았습니다. 에피소드별로는 성공한 시작 상태가 다르므로(assisted 0·3·7·11·21번, physical 0·1·6·7·13번) 실제로 다른 실행입니다.
- π0는 이 런타임에서 서버 예열 호출이 두 번 다 끊겨 돌지 못했습니다. 예열을 고친 뒤 27장의 런타임에서 같은 시작 상태로 돌렸습니다(위 표의 2/25). 실패 23번 중 21번은 마커를 집은 채 컵에 넣지 못했습니다.
  - 원인: 첫 호출은 컴파일 때문에 37 s 넘게 걸리고, 그동안 서버는 웹소켓 keepalive ping에 답하지 못합니다. websockets 기본값은 20 s마다 ping을 보내고 20 s 안에 답이 없으면 연결을 끊습니다. 그래서 첫 호출이 40 s 가까이 걸리면 예열 스크립트 쪽에서 연결이 끊깁니다(`keepalive ping timeout`).
  - 드라이버는 실패하면 서버를 새로 띄워 한 번 더 시도하지만, 새 서버는 다시 컴파일합니다. 이번에 π0.5는 두 번째 시도가 37 s로 통과했고, π0는 두 번 다 끊겼습니다. 23장의 첫 π0 실패도 같은 원인입니다(`logs/policy_eval_norobot/content/norobot_queue.log`).
  - 수정: `repro/policy_eval/ping_policy.py`가 자기 연결에서만 keepalive를 끕니다. 평가 중의 호출은 예열 뒤라 0.2 s 남짓이므로 그대로 둡니다. 27장의 런타임에서는 첫 호출이 π0.5 37.7 s, π0 39.9 s였고 둘 다 첫 시도에 통과했습니다.
- 결과는 Drive `results/policy_eval_ours/phys_colab/`(π0.5), `results/policy_eval_ours/cibphys_colab/out/marker_in_cup/`(π0), `results/policy_eval_ours/gr00t/out/marker_in_cup/s15_eval_gr00t_physical/`(GR00T)에 있습니다.
- 저자 메모의 마지막 항목 "손가락 최고 속도 낮추기"는 롤아웃으로 재 보았습니다(`repro/policy_eval/finger_speed.py`). 저장된 시뮬 상태에서 손가락 관절 위치를 읽으면, 닫기 명령 뒤 움직임의 95%를 마치는 데 중앙값 0.53 s(10–90%: 0.40–1.27 s), 열 때는 1.2 s가 걸립니다(π0.5·GR00T 기준 실행 50회, 닫기 215번·열기 183번). 실제 Robotiq 2F-85는 사양 최고 속도(150 mm/s)로 85 mm를 약 0.6 s에 닫으므로, 시뮬의 닫힘은 실물의 최고 속도와 비슷합니다. 그런데 이 손가락은 GAINS의 속도 제한(1.0 rad/s)이 아니라 자산 값(2.269 rad/s)으로 돌고 있었습니다. 그 원인과 제한을 걸고 다시 평가한 결과는 28장에 있습니다.
- 이로써 저자 장면 Marker in Cup에서 π0.5가 낮은 원인 후보에서 잡기 방식도 빠졌습니다. 빠진 것은 질량·카메라·지시문, 시작 격자·실행 지평, 톤매퍼, 배경 스캔의 로봇, 팔의 추적 지연, 충돌 메시의 입구, 잡기 방식입니다.

## 27. Cup in Bowl을 physical 잡기로 (2026-10-18)

25장의 저장소 설정 배치에서 세 정책은 π0 13/25, π0.5 20/25, GR00T 18/25였고, 논문 실제 값(88, 100, 68%)과의 r은 0.137이었습니다.

- 논문 시뮬 값(56, 92, 40%)과 견주면 π0(52%)와 π0.5(80%)는 가깝고, GR00T만 72%로 논문 시뮬 40%보다 높습니다. 실제로 가장 낮은 GR00T가 시뮬에서 π0를 앞지르는 것이 순위를 뒤집습니다.
- 논문의 시뮬 열을 이 세 정책으로만 계산해도 r은 0.935, MMRV는 0입니다. 정책이 셋뿐이라 생기는 문제는 아닙니다.
- 논문 Table G.2의 PolaRiS도 같은 모양입니다. GR00T N1.6이 Cup in Bowl에서 76%로 높게 나와 r이 −0.396입니다.
- GR00T의 실패 7번은 모두 컵을 집은 뒤 그릇에 넣지 못한 경우입니다. 실패 영상(4번)을 보면, 손가락 하나를 종이컵 안에 넣어 테두리를 집습니다. 그 뒤 쥔 컵으로 그릇을 책상 앞쪽으로 밀어내고 넣지 못합니다.
- 그래서 26.4의 저자 메모 항목 "assisted → physical"을 이 과제에도 시험했습니다. assisted에서는 닫히는 손가락에 닿은 물체가 고정 관절로 묶입니다(OmniGibson `manipulation_robot.py`, 관절이 없는 물체는 `FixedJoint`). 엉성하게 쥐어도 놓을 때까지 떨어지지 않으므로, 정책마다 다른 쥐기 품질의 차이를 지울 수 있습니다.

### 27.1 방법

- 같은 장면(`cup_in_bowl_ours_cfg`), 같은 시작 상태 25개, 같은 지시문·시간 제한(60 s)·지평(8)으로 `GRASPING_MODE=physical`만 바꿔 세 정책을 돌렸습니다. 세 평가 기록에 모두 `Changing grasping mode for robot0 from 'assisted' to 'physical'`이 남았습니다.
- GR00T N1.6은 이 머신에서, π0·π0.5는 Colab L4 한 대에서 돌렸습니다. 같은 L4에서 26.4에 빠졌던 π0의 저자 Marker in Cup physical도 같은 π0 서버로 동시에 돌렸습니다.
- 예열은 26.4에서 고친 `ping_policy.py`를 썼고, 두 서버 모두 첫 시도에 통과했습니다.
- 세 평가 모두 25회를 마치고 결과를 쓴 뒤, Isaac Sim 종료 중에 코드 139로 끝났습니다. 결과 파일은 온전합니다.
- L4는 약 2.9시간(18장의 단가로 약 4.5 CU) 썼고, 결과를 내려받아 확인한 뒤 종료했습니다.

### 27.2 결과

| 정책 | assisted (25장) | physical | 차이 (Fisher p) | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|---|
| π0 | 13/25 = 52% | **7/25 = 28%** | 0.15 | 56% | 88% |
| π0.5 | 20/25 = 80% | **19/25 = 76%** | 1.00 | 92% | 100% |
| GR00T N1.6 | 18/25 = 72% | **15/25 = 60%** | 0.55 | 40% | 68% |
| r / MMRV (대 논문 실제) | 0.137 / 0.133 | **0.189 / 0.133** | | 0.935 / 0 | |

- **잡기 방식도 Cup in Bowl의 원인이 아닙니다.** 세 정책 모두 physical에서 조금 내려갔지만 유의한 차이는 없습니다. 순위도 그대로입니다. 시뮬에서는 GR00T(60%)가 π0(28%)보다 높고(p = 0.045), 실제로는 π0(88%)가 GR00T(68%)보다 높습니다. r은 0.137에서 0.189로 거의 그대로이고, 95% 구간은 −0.37–0.72입니다.
- physical에서 세 정책은 각각 논문 시뮬 값과 통계적으로 구별되지 않습니다(π0 p = 0.085, π0.5 0.25, GR00T 0.26). 그러나 π0와 GR00T의 순서는 논문 시뮬과 반대입니다.
- 논문 시뮬에서도 이 두 정책의 차이(56 대 40%)는 25회씩으로는 유의하지 않습니다(p = 0.40). 그런데 이 두 값의 순서만 바꿔도 논문의 Cup in Bowl r은 0.907에서 0.661로, MMRV는 0.016에서 0.096으로 나빠집니다. 논문의 이 과제 r은 25회로 확정되지 않는 순서 하나에 크게 기대고 있습니다.
- 실패 양상:
  - π0: 실패 18번 중 11번은 컵에 닿지도 않았습니다(assisted 7번, p = 0.38). 잡기 방식은 처음 닿기 전에는 영향이 없으므로, 이 차이는 정책 샘플링의 흔들림입니다. 나머지 7번은 집은 뒤 넣지 못했습니다.
  - GR00T: 실패 10번 중 2번은 컵을 들지 못했습니다(assisted 0번). 영상(19번)을 보면 첫 접근에서 컵이 옆으로 넘어지고, 넘어진 컵을 여러 번 쥐었다가 놓칩니다. assisted였다면 닿은 채 닫는 순간 붙었을 경우입니다. 나머지 8번은 집은 뒤 넣지 못했습니다.
  - π0.5: 실패 6번 모두 컵을 집었습니다. 그중 3번은 컵이 한때 그릇 안에 들어갔지만, 로봇이 컵에서 떨어진 채 그릇 안에 있는 성공 조건까지는 가지 못했습니다.
- 저자 Marker in Cup physical의 π0는 2/25 = 8%였습니다(assisted 4/25, p = 0.67). 세 정책 physical로 r 0.872, MMRV 0.080입니다(26.4).
- 결과는 Drive `results/policy_eval_ours/cibphys_colab/`(π0·π0.5)와 `results/policy_eval_ours/gr00t/out/cup_in_bowl_ours_cfg/s15_eval_gr00t_physical/`(GR00T)에, 로그는 `logs/policy_eval_cibphys/`에 있습니다. 요약 폴더 `results_view/policy_eval/`에도 넣었습니다.
- 이로써 우리 Cup in Bowl에서 빠진 후보는 물체 질량(21장), 배치(25장), 잡기 방식(27장)입니다. 남은 차이는 π0와 GR00T의 순서 하나입니다. 저자 장면이 공개되지 않아, 컵·그릇의 크기와 모양 같은 장면 자체의 차이는 직접 비교할 수 없습니다.

## 28. 손가락 관절의 속도·힘 제한이 적용되지 않던 문제 (2026-10-19)

26.4에 적은 저자 튜닝 메모의 마지막 항목은 "손가락 최고 속도 낮추기(TBD: 어디선가 밖에서 다시 설정되는 것 같다. 매번 손으로 덮어써야 한다)"입니다. 따라가 보니 공개 코드에서 실제로 그런 일이 일어납니다.

- `PickPlaceTask._load`는 GAINS(`simfoundry/tasks/macros.py`)의 관절 최고 속도와 최대 힘을 씁니다. Robotiq 손가락에서 구동되는 두 관절(`left_outer_knuckle_joint`, `right_outer_knuckle_joint`)은 1.0 rad/s, 24입니다.
- OmniGibson은 이 두 값을 물리 뷰에만 쓰고 USD에는 남기지 않습니다(`joint_prim.py`의 `max_velocity` setter는 `set_max_velocities`만 부름). 시뮬레이터를 멈췄다 다시 켜면 물리 뷰를 USD에서 다시 만듭니다. 환경을 만들 때 한 번, 평가 단계가 `og.sim.stopped()`로 또 한 번 그렇게 합니다.
- Isaac Sim에서 확인했습니다. 저자 Marker in Cup 장면에서 환경을 만든 직후부터 두 손가락 관절은 자산 값 2.269 rad/s, 33이었고, 리셋 뒤에도, 멈췄다 켠 뒤에도 같았습니다. 팔 관절은 GAINS 값이 자산 값과 같아서 차이가 없습니다.
  - stiffness·damping도 멈췄다 켜면 10⁷이 되지만, `_reset_scene`이 리셋마다 다시 쓰므로 평가에는 GAINS 값이 들어갑니다. 최고 속도와 최대 힘만 다시 쓰는 곳이 없습니다.
- 롤아웃에서도 보입니다(`repro/policy_eval/finger_speed.py`). 지금까지의 실행에서 구동 손가락 관절의 에피소드별 최고 속도는 중앙값 1.67–1.74 rad/s, 최대 2.27–2.31 rad/s로 자산 값에 닿습니다. 지금까지의 모든 평가가 이 상태였습니다.

### 28.1 방법

- `PickPlaceTask`에 선택 인자 `reapply_joint_limits`를 넣었습니다. 켜면 리셋마다 gains 옆에서 GAINS의 최고 속도와 최대 힘을 다시 씁니다. 기본값 False는 예전과 같아서 앞 장들과 비교할 수 있습니다.
- 켜면 리셋 뒤에도, 멈췄다 켠 뒤의 리셋 뒤에도 두 손가락 관절이 1.0 rad/s, 24로 남는 것을 Isaac Sim에서 확인했습니다.
- `marker_in_cup_limits`: 저자 Marker in Cup 장면과 시작 상태 25개에 이 인자만 켭니다(`repro/policy_eval_ours/tasks/table_g1_marker_in_cup_limits.yaml`).
- π0.5는 Colab L4에서 같은 서버에 평가 두 개를 붙여 50회, π0는 25회 돌렸습니다. L4는 약 3.6시간(약 5.5 CU) 썼고, 결과를 내려받아 확인한 뒤 종료했습니다. GR00T N1.6은 이 머신에서 25회입니다.
- 이 머신의 GR00T 실행은 처음에 시작하자마자 두 번 실패했습니다. DreamZero 병합으로 `simfoundry/policies/dreamzero.py`가 `openpi_client.msgpack_numpy`를 읽게 됐는데, GR00T 실행이 쓰는 import 스텁에 그 모듈이 없었습니다. 이제 `run_local_gr00t.sh`가 필요한 모듈을 모두 갖춘 스텁을 직접 만듭니다.

### 28.2 결과

| 정책 | 기본 (18·19·26장) | 제한 적용 | 차이 (Fisher p) | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|---|
| π0 | 4/25 = 16% | **3/25 = 12%** | 1.00 | 40% | 40% |
| π0.5 | 55/100 = 55% | **25/50 = 50%** | 0.61 | 88% | 92% |
| GR00T N1.6 | 5/25 = 20% | **6/25 = 24%** | 1.00 | 28% | 28% |
| r / MMRV (대 논문 실제) | 0.961 / 0.080 | 0.890 / 0.080 | | | |

- **손가락 제한도 원인이 아닙니다.** 세 정책 모두 유의한 차이가 없고, π0.5는 여전히 논문 88%와 다릅니다(p = 0.002). 실패는 예전처럼 거의 모두 마커를 집은 채 컵에 넣지 못한 경우입니다.
- 제한은 실제로 걸렸습니다. 구동 관절의 에피소드별 최고 속도가 GR00T 25회 모두 1.00 rad/s, π0.5 50회 중앙값 1.00(최대 1.47) rad/s였습니다(기본 실행 중앙값 1.67–1.74, 최대 2.27–2.31). 닫기 명령 뒤 움직임의 95%를 마치는 시간의 중앙값은 π0.5에서 0.53 s에서 0.67 s로 늘었습니다.
- 이로써 26.4의 저자 메모 가운데 이 장면에서 시험할 수 있는 항목을 모두 시험했습니다. 색 보정은 톤매퍼(20.2), assisted→physical은 26.4와 27장, 손가락 속도는 이 장입니다. 메시 배경, 물체 마찰(평가 단계가 0.5), 자세·크기는 공개 장면에 이미 들어 있습니다. 실제 로봇의 동작을 재생해 궤적을 맞추는 항목은 실제 데이터가 없어 할 수 없습니다.
- 결과는 Drive `results/policy_eval_ours/limits_colab/`(π0·π0.5)와 `results/policy_eval_ours/gr00t/out/marker_in_cup_limits/`(GR00T)에, 로그는 `logs/policy_eval_limits/`에 있습니다.

## 29. 11단계 질량·마찰: 실측 질량으로 재고 고치기 (2026-10-19)

11단계(`11_make_objects_sim_ready.py`)는 물체마다 VLM에 사진, 방향 상자 크기, 부피를 주고 질량과 마찰을 한 번에 묻습니다. 그 값이 물체 USD의 질량과 이후 모든 단계의 마찰이 됩니다. 20장의 문구 수정도 실측값 없이 "종이컵 약 10 g" 같은 어림값으로 판단했습니다. 이 장은 실측 질량이 있는 물체로 먼저 재고, 그 결과로 11단계를 고칩니다.

### 29.1 벤치마크

- **YCB 물체 28개** (주 채점). 공개 스캔(google_16k)을 6단계 사진처럼 흰 배경, 3/4 시점으로 렌더하고(`repro/mass_friction/render_ycb.py`, 받기는 `ycb_fetch.sh`), 크기·부피·면적은 11단계처럼 잽니다. 정답은 YCB 논문(Calli 외, arXiv:1502.03143) Table II의 실측 질량입니다. 그릇 147 g, 머그 118 g, 접시 279 g, 가위 82 g, 스펀지 6.2 g, 토마토 캔 349 g 등입니다. YCB 과일은 플라스틱 모형(바나나 66 g)이라 뺐습니다. 사진으로는 진짜 과일과 구별되지 않습니다.
- **우리 장면 물체 8개** (보조). 6단계 사진과 8단계 메시(11단계 배율)를 그대로 씁니다. 정답은 일반적인 실물 값이나 저자 장면 값이라 대략입니다(종이컵 10 g, 마커 16 g, 가위 82 g, 펜 6 g, 바나나 120 g, 사과 130 g, 오렌지 140 g). 그릇·접시·쟁반·작은 과일 넷은 참고값이 불확실해 목록에만 둡니다.
- VLM은 파이프라인과 같은 Qwen3.5-9B(thinking 끔)이고 shim도 같습니다. temperature 0, seed 0입니다.
- 지표: 오차 배수 max(예측/정답, 정답/예측)의 중앙값, ±25% 안·2배 안 비율, 최악 배수. 관련 연구(NeRF2Physics)의 ALDE(평균 |ln 예측 − ln 정답|)와 MnRE(평균 min(예측/정답, 정답/예측))도 같이 냅니다. 토큰 한도에서 잘린 답은 실패로 셉니다(11단계에서는 예외로 단계가 멈춤).
- 스크립트: `repro/mass_friction/bench_mass.py`. 결과는 Drive `results/mass_friction/`에 있습니다.

### 29.2 비교한 방법 (물체마다 VLM 호출 한 번씩)

| 방법 | VLM에 주는 것 | VLM이 내는 것 | 질량 |
|---|---|---|---|
| original | 저장소 원래 11단계: 사진, 크기, 부피(메시 또는 상자의 65%) | 질량, 마찰 | 답 그대로 |
| current | 20장의 문구("속이 비면 질량은 부피의 일부"), 이음매를 합친 부피 | 질량, 마찰 | 답 그대로 |
| typical | 사진, 크기 | "이런 종류와 크기의 실제 물건은 보통 몇 g인가"(빈 것인지, 내용물이 든 것인지 포함) | 답 그대로 |
| material | 사진, 크기 | 재질, 밀도, 속이 찬 물체인지 얇은 껍질인지, 벽 두께, 내용물의 밀도와 채움 | 메시로 계산 |

- material의 계산: 속이 찬 물체는 밀도 × 메시가 감싸는 부피입니다. 껍질은 밀도 × 벽 두께 × 벽 면적입니다. 메시가 볼록 껍질의 40% 미만만 채우면 벽의 두 면이 다 복원된 컵·접시라 면적의 절반을, 더 차면 겉면만 있는 상자·캔이라 면적 전체를 씁니다. 내용물은 밀도 × 채움 × 부피를 더합니다. 작은 VLM에게 부피 계산을 맡기지 않고, 복원된 벽 두께(종이컵 약 3 mm, 실물 0.3 mm 안팎)에도 끌려가지 않게 하려는 것입니다. NeRF2Physics가 LLM에게 재질별 밀도와 두께를 받아 표면에서 적분하는 것과 같은 원리입니다.
- 조합 두 가지는 실행 전에 정했습니다. typical과 material의 기하평균, 그리고 material을 typical의 1/2–2배로 묶은 것입니다. 둘 중 하나만 답하면 그 값을 씁니다.

### 29.3 결과

**YCB 28개** (정답: 실측 질량)

| 방법 | 질량을 낸 물체 | 오차 배수 중앙값 | ±25% 안 | 2배 안 | 최악 | ALDE | MnRE |
|---|---|---|---|---|---|---|---|
| original | 26 | 2.07 | 29% | 43% | 7.4 | 0.790 | 0.540 |
| current | 25 | 1.48 | 32% | 64% | 7.2 | 0.544 | 0.657 |
| typical | 28 | 1.55 | 29% | 57% | 8.8 | 0.560 | 0.626 |
| material | 27 | 1.53 | 21% | 64% | 6.3 | 0.584 | 0.619 |
| 기하평균 | 27 | 1.43 | 32% | 71% | 5.0 | 0.447 | 0.685 |
| **material을 typical의 1/2–2배로** | **28** | **1.42** | 32% | **82%** | **4.8** | **0.440** | **0.689** |

**우리 장면 물체 8개** (정답: 대략값)

| 방법 | 오차 배수 중앙값 | ±25% 안 | 2배 안 | 최악 | ALDE | MnRE |
|---|---|---|---|---|---|---|
| original | 2.00 | 25% | 62% | 3.3 | 0.642 | 0.559 |
| current | 1.07 | 62% | 75% | 2.4 | 0.289 | 0.792 |
| typical | 1.53 | 12% | 75% | 3.0 | 0.540 | 0.613 |
| material | 1.37 | 50% | 75% | 2.4 | 0.378 | 0.722 |
| 기하평균 | 1.26 | 38% | 100% | 1.4 | 0.230 | 0.796 |
| **material을 typical의 1/2–2배로** | **1.23** | 50% | **100%** | **1.5** | **0.212** | **0.818** |

- **original**(저장소 원래 문구)이 가장 나쁩니다. 16장의 5.51 kg 그릇을 그대로 다시 냈고(같은 입력에 같은 답), 망치 2,740 g(실제 688 g), 주전자 850 g(178 g)처럼 크게 넘칩니다. 2개는 질량을 얻지 못했습니다. 그릇은 답이 토큰 한도에서 잘렸고, 테니스공은 답이 JSON 배열로 와서 11단계였다면 멈췄을 형식입니다.
- **current**(20장 문구)는 내용물이 든 통조림과 병을 빈 것으로 봅니다. 토마토 캔 50 g(349 g), 머스터드병 60 g(431 g)으로 약 7배 작습니다. 20장이 덧붙인 "속이 비면 질량은 부피의 일부"라는 문구 때문입니다. 3개(머그, 푸딩 상자, 참치 캔)는 같은 문장을 되풀이하다 잘렸습니다. 11단계였다면 그 물체에서 단계가 멈췄을 경우입니다. 우리 장면에서 성적이 좋은 것은 20장이 바로 이 물체들을 보며 문구를 고쳤기 때문입니다.
- **typical**은 크기를 덜 봅니다. 포크 300 g(34 g), 클램프 180·300 g(59·125 g)처럼 종류의 대표값을 냅니다.
- **material**은 밀봉된 통조림에 정확합니다(토마토 캔 343 g, 참치 캔 184 g, 고기 캔 399 g). 틀리는 곳은 VLM이 고른 값입니다. 판지 상자의 벽을 2.5–4 mm로 잡아 크래커 상자가 1,036 g(453 g)이 되고, 망치를 나무 밀도 하나(0.86)로 봐 223 g(688 g)이 되고, 테니스공을 펠트 껍질로 봐 12 g(58 g)이 됩니다. 얇은 숟가락은 스캔 부피가 실제보다 커서 126 g(30 g)입니다.
- 두 질문은 서로 다른 곳에서 크게 틀립니다. 그래서 material을 typical의 1/2–2배로 묶으면 큰 오차가 막힙니다. 이 조합과 기하평균은 실행 전에 정했습니다. 그래도 남는 큰 오차는 그릇 700 g(147 g, 법랑 강판 벽을 3.5 mm로 잡음), 망치 300 g, 포크 150 g, 크래커 상자 740 g입니다.
- 참고로 NeRF2Physics는 다른 데이터(ABO-500)에서 ALDE 0.771, MnRE 0.552이고, 같은 논문에서 VLM(LLaVA)에 질량을 직접 물은 값은 1.893, 0.306입니다.
- **이 결과는 모델에 따라 다릅니다.** Pipeline B는 Qwen3-VL-8B를 씁니다(16장: Qwen3.5는 cousin 목록을 자주 깨뜨림). 같은 벤치마크를 Qwen3-VL로 돌린 결과입니다(`results/mass_friction/bench_qwen3vl/`).

  | YCB 28 (Qwen3-VL) | 오차 배수 중앙값 | 2배 안 | 최악 | ALDE |
  |---|---|---|---|---|
  | current | 1.45 | 71% | 8.1 | 0.588 |
  | typical | 1.43 | 68% | 24.2 | 0.601 |
  | material | 1.98 | 50% | 8.6 | 0.791 |
  | 기하평균 | 1.39 | 71% | 5.2 | 0.513 |
  | material을 typical의 1/2–2배로 | 1.61 | 71% | 12.1 | 0.599 |

  Qwen3-VL은 "속까지 감싼 전체의 평균 밀도"라는 요청을 따르지 않고, 손잡이가 플라스틱인 공구에도 강철 밀도(7.85)를 줍니다. 드라이버 604 g(70 g), 가위 367 g(82 g), 숟가락 191 g(30 g)입니다. 이 모델로는 새 방식이 나아지지 않아, B4는 예전 방식을 기본으로 둡니다. 다른 VLM으로 바꿀 때는 `bench_mass.py`로 먼저 재야 합니다.
- 같은 물체·같은 사진·같은 문구라도 Ollama가 모델을 다른 조건에서 올리면 답이 바뀝니다. 장면을 다시 만들 때 빨간 마커의 material 답은 밀도 0.28(11.8 g)이었고, 벤치마크 때는 0.92(39 g)였습니다. 묶는 규칙이 두 경우 모두 실물(16 g)의 2배 안으로 둡니다.

### 29.4 고친 것

- `simfoundry/utils/mass_estimation.py`: 두 질문, 메시로 계산하는 질량, 묶는 규칙. 답이 잘리거나, JSON 배열로 오거나, 숫자가 "0.9–1.1"·"95%"처럼 오거나, JSON이 없어도 단계를 멈추지 않고 다른 질문의 답을 씁니다. 둘 다 실패하면 예전의 한 번 묻기로 돌아갑니다.
- 11단계가 기본으로 씁니다(`s11_sim.mass_method: material_typical`). `direct`로 두면 예전과 같습니다.
- cousin 단계(B4)도 같은 선택지를 받지만(`sim.mass_method`), 기본은 예전의 한 번 묻기(`direct`)로 두었습니다. 아래 Qwen3-VL 결과 때문입니다.
- 11단계 출력 `s11_sim/scene_objects_info.json`에 물체마다 `mass`와 `mass_estimate`(typical, material의 값과 답, 오류)를 더 적습니다. 원래 키는 그대로입니다.
- 벤치마크(`bench_mass.py`)가 이 모듈을 그대로 불러 씁니다. 그래서 벤치마크의 점수가 11단계가 쓰는 코드의 점수입니다.
- 장면을 다시 만들다 하나를 더 고쳤습니다. Serve Fruits의 초록 바나나에서 typical 답이 JSON 배열(`[{...}]`)로 와 단계가 멈췄습니다. 이제 배열의 첫 객체를 씁니다.

### 29.5 마찰

- **VLM의 마찰 답은 재질과 거의 무관합니다.** 벤치마크 40개 물체에서 세 질문 모두 0.2–0.7이고, 대부분 0.25–0.45입니다. 스펀지 0.35–0.45, 고무 골프공 0.35–0.45, 펠트 테니스공 0.35–0.5, 강철 렌치 0.3–0.7처럼 재질이 크게 달라도 값이 비슷합니다.
- **실측 마찰이 거의 없습니다.** NeRF2Physics가 공개한 마찰 측정은 6점(펠트 위 운동 마찰: 서랍 0.26, 회색 매트 0.92, 판지 0.36, 벽돌 0.54, 카펫 0.64, 책상 0.25)뿐이라 벤치마크로 쓰지 못했습니다. 그래서 마찰 값 자체는 고치지 않았고, 새 두 질문에서 "나무 책상 위에서 미끄러질 때의 계수"라고 무엇을 묻는지만 분명히 했습니다. 11단계는 material 답의 마찰을, 없으면 typical 답의 것을 씁니다.
- **대신 추정한 값이 시뮬레이터에 그대로 들어가게 했습니다.**
  - 12단계(PyBullet)는 두 물체의 마찰을 곱합니다. 바닥(`plane.urdf`)이 1.0이라 물체의 추정치가 그대로 들어갑니다. 상자로 재 보니 0.2/0.5/0.9가 유효 마찰 0.205/0.523/0.970이었습니다.
  - PhysX(14단계 이후)는 기본적으로 두 재질의 평균을 씁니다. 바닥 재질이 0.5라 추정치가 희석됐습니다(0.9 → 0.7, 0.2 → 0.35).
  - `set_floor_material`(`simfoundry/utils/og_utils.py`)이 바닥 충돌체에 마찰 1.0, 결합 방식 multiply인 재질을 붙입니다. PhysX는 두 재질의 결합 방식 중 우선순위가 높은 것을 쓰므로(average < min < multiply < max), 물체와 바닥 사이는 추정치 그대로, 물체끼리는 예전처럼 평균입니다. 반발은 바닥 기본값(0.8, 평균)을 그대로 둡니다.
  - `set_obj_materials`가 이 함수를 부르므로 14단계, 텔레옵(2, 2b), B6에 모두 적용됩니다. 평가 단계는 이 함수를 부르지 않아 예전 그대로입니다(물체 0.5, 손가락 2.0, Table G.1 조건).
- **Isaac Sim 확인** (`repro/friction/check_friction.py`, `put_cup_in_bowl_v2_mass`, 결과 `diagnostics/friction/put_cup_in_bowl_v2_mass.json`). 그릇 사본을 0.5 m/s로 밀었습니다.

  | 물체 마찰 | 예전 바닥: 유효 마찰 / 미끄러진 거리 (예상) | 새 바닥: 유효 마찰 / 미끄러진 거리 (예상) |
  |---|---|---|
  | 0.2 | 0.35 / 3.48 cm (3.64) | **0.2** / 6.36 cm (6.37) |
  | 0.5 | — | 0.5 / 2.33 cm (2.55) |
  | 0.9 | 0.7 / 1.57 cm (1.82) | **0.9** / 0.94 cm (1.42) |

  - 낮은 마찰에서는 예상과 같습니다. 높은 마찰에서는 그릇이 순수 미끄럼 계산보다 빨리 멈춥니다. 20.4의 예전 바닥 시험에서도 유효 마찰 0.75일 때 예상의 79%였으니, 이 물체가 높은 마찰에서 보이는 성질입니다.
  - 같은 실행에서 PhysX의 질량이 11단계 추정치와 같았습니다(컵 0.015, 펜 0.0045, 그릇 0.2932 kg). 가라앉힌 뒤 100스텝 동안 움직임은 0.01 mm 이하였습니다.

### 29.6 장면 다시 만들기

세 장면을 새 이름(`put_cup_in_bowl_v2_mass`, `put_marker_in_cup_v2_mass`, `serve_fruits_mass`)으로 복사해 11–14단계를 다시 돌렸습니다. 11단계가 자산 이름을 장면 이름에서 만들므로 기존 USD는 건드리지 않습니다. 세 장면 모두 14단계까지 끝났습니다.

| 장면 | 물체 | 예전 (USD) | 새로 | 실물·저자 값 |
|---|---|---|---|---|
| Cup in Bowl v2 | 주황 그릇 (지름 26 cm) | 500 g | 293 g | 수백 g |
| | 검은 펜 | 10 g | 4.5 g | 약 6 g |
| | 흰 종이컵 | 70 g | 15 g | 약 10 g |
| Marker in Cup v2 | 파란 가위 | 50 g | 76.5 g | 약 82 g |
| | 녹색 무늬 종이컵 | 150 g | 12.5 g | 약 10 g |
| | 빨간 마커 | 37.6 g | 11.8 g | 약 16 g |
| Serve Fruits | 바나나 | 610 g | 135 g | 저자 120 g |
| | 사과 | 123 g | 137.5 g | 저자 130 g |
| | 오렌지 | 162 g | 138 g | 저자 140 g |
| | 배("orange tangerine") | 137 g | 100 g | 저자 240 g (우리 것이 작게 복원됨) |
| | 주황 접시 | 90 g | 36 g | 저자 400 g (실물은 두 장을 겹친 것) |
| | 청록 쟁반 | 30 g | 180 g | 저자 500 g |
| | 유령 바나나 둘 (24장) | 363·253 g | 144·105 g | — |

- 종이컵 두 개가 실물(약 10 g)의 7–15배에서 1.25–1.5배로 왔습니다. 바나나의 610 g도 사라졌습니다.
- 주황 접시는 오히려 멀어졌습니다. typical 답(18 g)이 너무 작아, 묶는 규칙이 material 답(88 g)을 36 g으로 끌어내렸습니다. 묶는 규칙은 typical이 크게 틀린 경우를 막지 못합니다. 규칙은 벤치마크 전에 정한 그대로 두었습니다.
- 12단계(PyBullet 안정화)와 14단계(OmniGibson 가라앉힘)는 세 장면 모두 정상으로 끝났고, 14단계 로그에 새 마찰이 찍혔습니다.
- 이 장면들로 정책 평가는 다시 하지 않았습니다. 21장에서 질량을 실물에 가깝게 고쳐도 성공률이 유의하게 바뀐 정책이 없었습니다.

### 29.7 관련 연구

- **image2mass**(Standley 외, CoRL 2017): 상품 사진·크기·무게로 학습한 질량 추정. 3D 형상 추정을 쓰는 모델이 기준선보다 낫고 사람과 비슷합니다. 사진만으로는 장난감과 진짜를 가릴 수 없다고 지적합니다(YCB 모형 과일을 뺀 이유).
- **NeRF2Physics**(Zhai 외, CVPR 2024): LLM이 물체마다 재질 후보와 밀도 범위, 재질별 두께를 내고, 표면 점마다 면적 × 두께 × 밀도를 더합니다. ABO-500에서 ALDE 0.771이고, VLM에 질량을 직접 물은 기준선은 1.893으로 가장 나쁩니다. 이 장의 material 질문과 같은 원리입니다. PUGS(2025)와 PhysGS(2025)는 이를 Gaussian splatting으로 옮겨 개선했습니다.
- **PhysObjects**(Gao 외, ICRA 2024)는 내용물·밀봉 여부·질량 같은 물리 개념으로 VLM을 미세조정했고, **PhysBench**(ICLR 2025)는 VLM 75개가 물리 속성 이해에 약하다고 보고합니다. 20장 문구가 통조림을 빈 것으로 본 것, 작은 VLM의 직접 답이 불안정한 것과 같은 방향입니다.
- 마찰은 재질을 알아본 뒤 실측 마찰표에서 찾는 방식이 표준입니다(Brandão 외, 이족 로봇의 미끄러운 바닥 걷기).

### 29.8 한계

- 실측 정답은 YCB 28개뿐입니다. ABO-500(77 GB)으로 더 넓게 재지는 않았습니다.
- 남은 큰 오차는 VLM이 고르는 벽 두께와 밀도입니다(법랑 그릇, 망치, 판지 상자). NeRF2Physics처럼 재질 후보와 범위를 받거나 내용물·밀봉 여부를 따로 묻는 개선은, 이 28개로 다시 맞추면 과적합이라 하지 않았습니다. 새 정답 자료가 있어야 합니다.
- 마찰 값 자체는 실측 자료가 없어 고치지 못했습니다. 시뮬레이터에 추정치가 그대로 들어가게만 했습니다.
- 논문 설정인 Gemini는 키가 없어 재지 못했습니다.

### 29.9 파일과 다시 실행

| 파일 | 내용 |
|---|---|
| `simfoundry/utils/mass_estimation.py` | typical·material 질문, 메시로 계산하는 질량, 묶는 규칙, 답 해석 |
| `scripts/pipeline/A_reconstruction/stages/11_make_objects_sim_ready.py` | `s11_sim.mass_method`(기본 `material_typical`), 출력에 `mass`·`mass_estimate` |
| `scripts/pipeline/B_augmentation/stages/4_make_cousins_sim_ready.py` | `sim.mass_method`(기본 `direct`) |
| `simfoundry/utils/og_utils.py` | `set_floor_material`, `set_obj_materials`가 바닥 재질도 붙임 |
| `repro/mass_friction/ycb_fetch.sh`, `render_ycb.py`, `bench_mass.py` | YCB 받기·렌더, 질량 벤치마크 |
| `repro/friction/check_friction.py` | 바닥 결합 방식을 따르는 미끄럼 시험, PhysX 질량 확인 |

```bash
# YCB 28개를 받아 렌더 (GPU 필요 없음)
N="003_cracker_box 004_sugar_box ... 061_foam_brick"   # bench_mass.py의 YCB 목록
repro/mass_friction/ycb_fetch.sh ~/.cache/ycb_google_16k $N
PYOPENGL_PLATFORM=egl python -I repro/mass_friction/render_ycb.py ~/.cache/ycb_google_16k results/mass_friction/ycb $N
# 벤치마크 (Ollama, GPU가 빈 뒤에)
SIMFOUNDRY_LOCAL_BACKEND=1 SIMFOUNDRY_OLLAMA_MODEL=qwen3.5:9b SIMFOUNDRY_OLLAMA_THINK=0 \
  python repro/mass_friction/bench_mass.py results/mass_friction/bench --methods original,current,typical,material
# 장면 사본으로 11–14단계, 그리고 미끄럼 시험
repro/run_local_pipeline.sh A_reconstruction --scene-name put_cup_in_bowl_v2_mass --include 11,12,13,14 --no-stream
PYTHONPATH=$PWD python repro/friction/check_friction.py --scene-dir Data/put_cup_in_bowl_v2_mass --slide-object iter_0 \
  --out results/diagnostics/friction/put_cup_in_bowl_v2_mass.json
```

## 30. 손목 카메라 확인과 로컬 π0·π0.5 평가 (2026-10-19)

이 장에는 세 가지가 있습니다. 먼저 손목 카메라의 렌즈를 다시 확인했고, 앞서 의심한 불일치는 없었습니다. 다음으로 17장의 3DGS 배경을 로봇 손목 카메라 시점에서 확인했습니다. 마지막으로 Colab을 더 쓰지 않기로 해서 π0·π0.5를 이 머신(RTX 3060 12 GB)에서 돌리게 했고, 지금까지 찾은 수정을 모두 넣은 Cup in Bowl 장면을 세 정책으로 평가했습니다.

### 30.1 손목 카메라의 렌즈: 불일치 없음

- 장면 파일에는 손목 카메라의 렌즈 값이 없어서, 장면을 열면 OmniGibson 기본값(초점 17 mm, 조리개 폭 20.995 mm, 가로 63°)이 들어갑니다. 이 값만 보고 손목 화면이 DROID 손목 카메라(ZED Mini)보다 좁다고 의심했습니다. DROID `intrinsics.json`의 에피소드 72,467개에서 1280×720 기준 fx 중앙값은 731.8 px(가로 82°)입니다.
- 그런데 평가 단계는 장면을 연 뒤 로봇의 모든 카메라를 초점 2.8 mm, 조리개 폭 5.376 mm로 바꿉니다(`1_eval_policy_og_scene.py:404-410`). 그러면 가로 87.7°, 1280 폭에서 fx 666.7 px입니다. 정책이 보는 손목 화면은 DROID보다 약 9% 넓습니다. 좁지 않습니다.
- 그래서 손목 렌즈를 12 mm로 바꾼 장면 사본으로 π0.5를 돌리려던 계획은 취소했습니다. 그 사본도 평가 단계가 덮어써서 기본 장면과 같았을 것입니다. 사본과 그 스크립트는 지웠습니다.
- 같이 확인한 것:
  - openpi의 DROID 정책(π0·π0.5)은 외부 카메라 중 `exterior_image_1_left` 하나와 손목 화면만 읽습니다(openpi `src/openpi/policies/droid_policy.py`). 두 번째 외부 카메라는 영상에만 남습니다.
  - 그리퍼 상태 입력. 평가 단계는 마지막 두 관절의 평균을 구동 관절 상한(0.785 rad)으로 나누고 2를 곱합니다. 둘 중 하나는 범위가 ±0.02 rad인 바깥 손가락 관절이라 거의 0이므로, 결과는 구동 관절 각도의 비율이 됩니다. 롤아웃에서 읽으면 열림 0, 마커를 쥐면 0.72, 다 닫으면 약 1.0으로 DROID의 0(열림)–1(닫힘) 규약과 맞습니다.
  - PolaRiS용 π0.5. 버킷에는 시뮬 데이터 10%를 섞어 1,000단계 미세조정한 `checkpoints/polaris/pi05_droid_jointpos_polaris`가 따로 있습니다. 논문 부록(`supp/A-realtosimevals.tex`)은 이 체크포인트가 시뮬에서 더 높았지만, 정책을 시뮬레이터에 맞추지 않는다는 원칙에 따라 비교에서 뺐다고 적습니다. Table G.1은 기본 체크포인트의 값이고, 우리도 같은 것을 썼습니다(18.2).

### 30.2 3DGS 배경을 손목 카메라에서 보기

- 질문: 17장의 3DGS 배경은 로봇 외부 카메라 시점에서 무너졌는데(17.4, 17.7), 로봇 팔에 달린 손목 카메라에서는 어떤가.
- 방법
  - `repro/scripted_pick_place.py --wrist`가 스크립트 집기 시연 동안 손 카메라를 따로 녹화합니다(`<out>.wrist.mp4`). 렌즈는 평가 단계와 같은 2.8 mm / 5.376 mm(87.7°)이고, 해상도는 640×360입니다.
  - 17.4의 기본 배경과 17.7의 sreg+carve 배경에서 두 장면을 같은 동작으로 돌렸습니다. 네 번 모두 과제에 성공했습니다(Cup in Bowl 699스텝, Marker in Cup 857스텝). 배경만 다르고 동작은 같으므로 같은 스텝은 같은 팔 자세입니다.
  - 이 장면들의 로봇은 Panda 그리퍼입니다. 카메라가 손에 붙은 위치는 DROID의 Robotiq 장착과 조금 다릅니다.
- 그림: Drive `results/wrist_3dgs/{cup_in_bowl,marker_in_cup}_wrist_sheet.jpg`(`repro/g4/wrist_sheet.py`). 행은 동작의 8–92% 시점이고, 열은 배경마다 외부 카메라(teleop 배치의 첫 카메라)와 손목 카메라입니다. 영상도 같은 폴더에 있습니다.

| 배경 | 손목 카메라가 책상을 내려다볼 때 | 손목 카메라가 옆을 볼 때 (Marker in Cup에서 마커를 세운 뒤) |
|---|---|---|
| 17.4 기본 | 무너짐. 갈색·보라·청록 얼룩, 화면을 가로지르는 검은 바늘 모양, 그릇 안에 떠 있는 갈색 파편 | 무너짐. 흰 얼룩과 무지개색 줄 |
| 17.7 sreg+carve | 흐리지만 나무 책상으로 보임. 결 방향 줄무늬가 있고, 책상 가장자리 쪽에 흰 띠와 검은 파편이 보임. 그릇 안은 깨끗함 | 벽은 흐린 베이지 얼룩이고, 왼쪽에 큰 검은 덩어리가 있음 |

- **답: 17.4의 기본 배경은 손목 카메라에서도 깨집니다. 17.7의 sreg+carve는 손목 카메라가 책상을 내려다보는 동안에는 알아볼 만하지만, 옆을 보면 무너집니다.**
  - 손목 카메라가 내려다보는 책상 윗면은 서서 찍은 휴대폰 영상이 위에서 본 곳입니다. 그래서 책상 높이에서 옆으로 보는 외부 카메라보다 학습 시점에 가깝습니다.
  - 그래도 손목 카메라는 책상에서 10–30 cm 위라 영상보다 훨씬 가깝습니다. 그래서 질감이 흐리고 가우시안의 결이 줄무늬로 보입니다.
- 정책 평가에 미치는 영향: π0·π0.5는 외부 카메라 한 대와 손목 화면을 함께 봅니다. 외부 카메라 화면이 무너지는 한(17.7), 손목 화면이 낫더라도 이 배경으로 평가할 수는 없습니다. Table G.1 평가(18장 이후)는 3DGS를 거치지 않습니다. 저자 DROID 장면 5개와 우리 복원 장면은 모두 메시 배경(`droid_v1.usd`, Stack Dishware는 `droid_v2.usd`)을 씁니다.

### 30.3 π0·π0.5를 이 머신에서

Colab을 더 쓰지 않기로 해서(2026-10-19), Colab에서 돌리던 π0·π0.5 평가를 이 머신으로 옮겼습니다. 에피소드는 Colab 때와 같은 `run_table_g1.sh`의 평가 함수로 돌리므로 평가 단계, 설정, 출력 구조가 같습니다. 다른 것은 경로와 서버를 띄우는 방법입니다(`repro/policy_eval_ours/run_local_openpi.sh`).

- openpi는 Colab과 같은 커밋 `15a9616`을 `/home/inner/real2sim/tools/openpi`에 uv 환경으로 두었습니다. 서버는 그 환경의 python으로 직접 띄웁니다. `uv run`은 띄울 때마다 환경을 다시 맞추려 하기 때문입니다. simfoundry 환경은 `openpi_client`를 설치하지 않고 openpi 소스에서 읽습니다(`PYTHONPATH`).
- 체크포인트(π0.5 12.4 GB, π0 12.0 GB)는 공개 버킷에서 익명으로 받았습니다.
  - openpi의 내려받기(gcsfs)는 11 GB 근처에서 두 번 멈췄습니다. 연결은 열려 있는데 바이트가 오지 않았습니다.
  - `repro/policy_eval_ours/fetch_openpi_ckpt.py`는 gcsfs로 목록만 받고, 파일마다 curl로 받습니다. 끊긴 곳에서 이어받고, 30 s 동안 100 kB/s 아래면 끊고 다시 받습니다. π0을 받을 때도 한 번 멈췄고, 이어받아 끝났습니다.
- **메모리.** 12 GB에 정책 서버와 Isaac Sim이 같이 올라가야 합니다.
  - JAX는 기본으로 카드의 75%를 미리 잡습니다. 비율을 0.6(7.3 GB)으로 줄이면 서버는 뜨지만, Isaac Sim이 첫 화면에서 장치 메모리 부족으로 죽었습니다. 0.52(6.4 GB)에서는 π0.5가 올라가다가 1 GB짜리 배열(임베딩 표)을 잡지 못했습니다.
  - JAX의 platform 할당기(`XLA_PYTHON_CLIENT_ALLOCATOR=platform`, 미리 잡지 않음)는 모델을 올리며 쓰고 버린 메모리를 돌려줍니다. 그러면 π0.5 서버는 예열 뒤 6.6 GB를 쥐고, 호출 시간은 0.34 s로 고정 비율일 때와 같습니다. 스크립트의 기본값입니다.
  - 평가 쪽은 두 번째 외부 카메라와 깊이 출력을 뺐습니다(`nv_franka_droid_single_cam`: 첫 외부 카메라를 같은 자세·렌즈로 두고 rgb만). π0·π0.5는 그 카메라를 읽지 않으므로(30.1) 정책 입력은 같고, 저장되는 영상에서만 빠집니다.
  - 이렇게 하면 서버와 평가를 합쳐 10.9–11.1 GB입니다.
- **속도.** π0.5로 Cup in Bowl 25회(성공 22번)가 Isaac Sim 시작을 포함해 18분 걸렸습니다. 시뮬레이션은 초당 7.6–9.3스텝(제어 15 Hz)으로 돕니다.
- GPU는 다른 세션과 같이 씁니다. 그래서 서버는 다른 세션의 파이프라인 단계나 VLM 재생이 없고, Ollama에 올라간 모델이 없고, GPU 메모리가 1 GB 미만일 때만 띄웁니다.
- 사용자가 허락한 두 번째 머신 super2(같은 RTX 3060 12 GB)에서도 같은 스크립트로 돌렸습니다(30.6부터).
  - 그곳의 SimFoundry 작업 사본은 다른 세션이 쓰고 있어서 건드리지 않았습니다. 이 저장소의 추적 파일과 필요한 장면만 별도 폴더(`~/real2sim/claude-eval-54d9e2`)에 두고, `deps`는 그 작업 사본의 것을 읽기만 합니다(같은 OmniGibson 커밋 `d89aae4e`).
  - openpi 환경과 체크포인트는 이 머신에서 같은 경로로 복사했습니다(약 32 GB, 100 MB/s).
  - 결과는 이 머신의 `results/`로 가져와 함께 집계합니다.

### 30.4 Cup in Bowl: 지금까지의 수정을 모두 넣은 장면

- 장면 `cup_in_bowl_ours_real`은 21장의 `cup_in_bowl_ours_mass`(그릇이 논문 사진처럼 책상의 검은 사각형 위, 시작 상태 25개도 같음)에 지금까지의 수정을 모두 넣은 것입니다.
  - 질량: 29장의 방법으로 11단계가 낸 값(컵 15 g, 그릇 293 g, 펜 4.5 g)
  - 손가락 관절의 속도·힘 제한을 매 리셋마다 적용(28장, 태스크 `ours_cup_in_bowl_limits`)
  - physical 잡기(26.4)
- 세 정책 모두 25회씩 두 번 돌렸고, 순위를 가르는 π0와 GR00T는 두 번 더 돌렸습니다. GR00T N1.6은 19장과 같은 방법, π0·π0.5는 30.3의 방법입니다(π0의 반복 3·4는 super2).

| 정책 | 25회씩 | 합계 | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|
| π0 | 18, 19, 15, 20 | **72/100 = 72%** | 56% | 88% |
| π0.5 | 22, 24 | **46/50 = 92%** | 92% | 100% |
| GR00T N1.6 | 19, 17, 21, 17 | **74/100 = 74%** | 40% | 68% |

| Cup in Bowl 장면 | r | r 95% 구간 | MMRV |
|---|---|---|---|
| 19장 우리 복원 | −0.143 | −0.78..+0.57 | 0.173 |
| 21장 질량 수정 | 0.446 | −0.15..+0.98 | 0.133 |
| 25장 저장소 설정의 배치 | 0.137 | −0.80..+0.94 | 0.133 |
| 27장 저장소 설정의 배치 + physical 잡기 | 0.189 | −0.37..+0.72 | 0.133 |
| 30장 모든 수정, 처음 50회씩 | 0.839 | +0.19..+1.00 | 0.000 |
| **30장 모든 수정, π0·GR00T 100회씩** | **0.726** | **+0.21..+0.99** | **0.133** |
| 논문 (같은 세 정책) | 0.935 | | 0.000 |
| 논문 (다섯 정책, Table G.1) | 0.907 | | 0.016 |

- **수정을 모두 넣어도 Cup in Bowl은 논문과 맞지 않습니다.**
  - 처음 50회씩에서는 세 정책의 순위가 실제와 같았습니다(π0 37/50, GR00T 36/50. r 0.839, MMRV 0). 그러나 순위를 가른 차이가 1회였습니다.
  - 두 정책을 50회씩 더 돌리자 π0 72/100, GR00T 74/100으로 뒤집혔습니다. 우리 시뮬에서 두 정책은 같습니다(p = 0.87, 차이의 95% 구간 −14..+10%p).
  - 실제의 차이(π0가 20%p 높음)도, 논문 시뮬의 차이(16%p)도 이 구간 밖입니다. 처음 순위가 맞은 것은 우연이었습니다.
  - r은 0.726으로 21장(0.446)보다 높지만, π0와 GR00T의 순위가 실제와 반대라 MMRV는 0.133입니다.
- 정책별로 보면:
  - π0.5는 논문 시뮬과 같은 92%입니다.
  - π0는 72%로 논문 시뮬 56%와 실제 88% 사이입니다(시뮬과 p = 0.15).
  - GR00T는 74%로 논문 시뮬 40%보다 높습니다(p = 0.002). 논문 Table G.2의 PolaRiS에서도 GR00T는 76%였습니다(27장). 우리 시뮬의 GR00T는 SimFoundry 장면보다 PolaRiS 쪽처럼 행동합니다.
  - 남은 어긋남은 GR00T입니다. 저자 Cup in Bowl 장면이 공개되지 않아, 어느 물체나 배치가 GR00T를 낮추는지 22.4처럼 바꿔 끼워 볼 수 없습니다.
- 실패 양상(처음 50회씩):
  - π0.5의 실패 4번은 모두 컵을 집은 뒤입니다. 컵을 떨어뜨려 눕힌 것, 그릇을 쳐서 넘어뜨린 것, 컵을 쥔 채 그릇을 찾지 못한 것이었습니다.
  - π0의 13번: 컵에 닿지도 않음 5, 닿았지만 못 집음 3, 집은 뒤 못 넣음 5
  - GR00T의 14번: 닿았지만 못 집음 6, 집은 뒤 못 넣음 8
- 결과는 Drive `results/policy_eval_ours/openpi_local/out/cup_in_bowl_ours_real/`(π0·π0.5), `results/policy_eval_ours/gr00t/out/cup_in_bowl_ours_real/`(GR00T)에 있습니다. 두 반복을 합친 집계와 판정표는 `diagnostics/policy_eval_ours/ours_real/`에 있습니다.

```bash
GRASPING_MODE=physical repro/policy_eval_ours/run_local_gr00t.sh all 25 main repro/policy_eval_ours/ours.yaml:cup_in_bowl_ours_real
repro/policy_eval_ours/run_local_openpi.sh server pi05
GRASPING_MODE=physical repro/policy_eval_ours/run_local_openpi.sh all pi05 25 main repro/policy_eval_ours/ours.yaml:cup_in_bowl_ours_real
repro/policy_eval_ours/run_local_openpi.sh stop pi05   # π0도 같은 방법, 반복은 태그 rep2
python repro/policy_eval_ours/correlate.py --summary results/diagnostics/policy_eval_ours/ours_real/cup_in_bowl_ours_real_pooled.json --draws 20000
```

### 30.5 저자 장면 Marker in Cup의 π0.5: 이번에 더 확인한 것

26장까지 원인 후보를 여럿 지웠지만, 저자 장면 Marker in Cup의 π0.5는 55%로 논문 88%와 다릅니다. 이번에 더 확인한 것은 셋입니다.

- 손목 카메라 렌즈와 그리퍼 상태 입력은 맞습니다(30.1).
- 마커를 놓는 위치가 한쪽으로 치우치지 않습니다.
  - 기준 실행 100회(18.4)의 롤아웃에서, 마커를 들어 옮긴 뒤 처음 놓은 순간 머그 중심에 대한 마커 중심의 위치를 쟀습니다. x는 로봇에서 멀어지는 방향, y는 로봇의 왼쪽입니다.
  - 성공 42번은 거리 중앙값 4.3 cm, 평균 (x +3.0, y −1.0) cm입니다. 실패 29번은 7.2 cm, 평균 (x +1.2, y −4.2) cm이고 표준편차는 5–7 cm입니다.
  - 카메라나 손목 장착이 어긋나 늘 한쪽으로 놓는 모양이 아니라, 놓는 위치가 넓게 흩어집니다.
- **저자 장면은 공개되지 않은 OmniGibson으로 저장됐습니다.**
  - 장면 JSON의 `versions`는 Marker in Cup·Serve Fruits가 OmniGibson 3.8.0 커밋 `da74f039`, 나머지 셋이 `a0ca8378`입니다. 두 커밋 모두 공개 BEHAVIOR-1K 저장소에 없습니다(GitHub API가 찾지 못함).
  - 우리는 저장소 설치 스크립트가 고정한 `d89aae4e`(2026-03-05)에 마찰 저장 패치를 얹어 Isaac Sim 5.1.0으로 씁니다.
  - 그 사이의 물리·렌더 차이는 확인할 수 없습니다. 남은 차이의 후보입니다.

### 30.6 Marker in Cup: 같은 수정을 넣은 우리 장면

- 장면 `marker_in_cup_ours_real`은 21장의 `marker_in_cup_ours_mass`에 30.4와 같은 수정을 넣은 것입니다.
  - 질량: 11단계가 낸 값(마커 11.8 g, 컵 12.5 g. 21장은 18 g, 15 g)
  - 손가락 제한(`ours_marker_in_cup_limits`)과 physical 잡기
  - 장면 파일은 `args.mass` 두 값만 다르고, 시작 상태 25개도 같습니다.
- π0·π0.5는 super2에서, GR00T N1.6은 이 머신에서 25회씩 두 번 돌렸습니다.

| 정책 | 25회 ×2 | 합계 | 21장 (assisted 잡기) | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|---|
| π0 | 6, 4 | **10/50 = 20%** | 8/25 = 32% | 40% | 40% |
| π0.5 | 13, 16 | **29/50 = 58%** | 20/25 = 80% | 88% | 92% |
| GR00T N1.6 | 5, 4 | **9/50 = 18%** | 8/25 = 32% | 28% | 28% |

- **순위와 상관은 논문과 맞습니다.** r 0.991(95% 구간 +0.87..+1.00), MMRV 0입니다. 논문은 0.995, 0.008입니다.
- **성공률은 세 정책 모두 21장보다 낮습니다.** π0.5 80→58%(p = 0.074), π0 32→20%(p = 0.27), GR00T 32→18%(p = 0.24)입니다. π0.5는 논문 시뮬 88%와 다릅니다(p = 0.009).
- 실패는 대부분 마커를 집은 뒤 컵에 넣지 못한 경우입니다. π0 40번 중 34번, π0.5 21번 중 18번, GR00T 41번 중 40번입니다.
- 바뀐 것 가운데 physical 잡기는 저자 장면에서도 π0.5를 낮췄습니다(26.4: 55→44%). 그래서 다른 수정은 그대로 두고 잡기만 assisted로 돌려 비교했습니다(π0.5, 같은 시작 상태, 25회 ×2, 이 머신).
  - 결과는 17, 17로 **34/50 = 68%**입니다. physical의 58%와 구별되지 않고(p = 0.41), 21장의 80%와도 구별되지 않습니다(p = 0.41).
  - 따라서 80%에서 58%로 내려간 것을 어느 한 수정 탓으로 볼 수 없습니다. 각 차이(약 10%p)는 50회로 가려낼 수 있는 크기보다 작습니다.
  - 잡기가 assisted이면 성공한 에피소드가 더 빨랐습니다(스텝 중앙값 281, physical은 438–497). 실패 16번은 모두 마커를 집은 뒤 컵에 넣지 못한 경우입니다.
- 결과는 Drive `results/policy_eval_ours/openpi_local/out/marker_in_cup_ours_real/`(π0·π0.5), `results/policy_eval_ours/gr00t/out/marker_in_cup_ours_real/`(GR00T)에 있습니다. 집계와 판정표는 `diagnostics/policy_eval_ours/ours_real/`에 있습니다.

### 30.7 Cup in Bowl: 저자 그릇으로 바꿔 보기

- 30.4에서 남은 어긋남은 GR00T가 논문 시뮬(40%)보다 높다는 것입니다. 22.4에서는 방해물 접시 하나를 저자 것으로 바꾸자 GR00T가 52%에서 16%로 떨어졌습니다.
- Cup in Bowl은 저자 장면이 공개되지 않았습니다. 대신 저자의 YAM `bowls` 장면(공개 자산)에 주황 그릇이 두 개 있습니다.
  - 큰 그릇 `ubzmrk`(26×25×11 cm)은 논문 Cup in Bowl 사진의 그릇과 색(붉은 주황)과 높이/너비 비(0.43)가 맞습니다.
  - 우리 그릇(26×26×14 cm)은 0.53이고 더 노랗습니다.
- `cup_in_bowl_ours_real_authbowl`은 30.4의 장면에서 그릇만 이것으로 바꾼 장면입니다(`build_droid_scene.py`의 `from_scene`·`from_name`).
  - 질량은 우리 값(293 g)으로 두었습니다.
  - 시작 상태는 에피소드마다 그릇의 xy를 그대로 두고, 높이와 자세만 새 그릇에 맞췄습니다(`swap_init_object.py`).
  - 첫 에피소드 영상에서 그릇이 검은 사각형 위에 제대로 놓인 것을 확인했습니다.

| 정책 | 우리 그릇 (30.4) | 저자 그릇 (25회 ×2) | 차이 (Fisher p) | 논문 시뮬 |
|---|---|---|---|---|
| π0 | 72/100 = 72% | 19, 14 → **33/50 = 66%** | 0.46 | 56% |
| GR00T N1.6 | 74/100 = 74% | 19, 18 → **37/50 = 74%** | 1.00 | 40% |
| π0.5 | 46/50 = 92% | 24, 24 → **48/50 = 96%** | 0.68 | 92% |

- **그릇은 원인이 아닙니다.** GR00T는 저자 그릇에서도 74%로 그대로이고, 논문 시뮬 40%와 다릅니다(p = 0.006). 세 정책으로 낸 r은 0.600(95% 구간 +0.15..+0.95), MMRV는 0.133입니다. π0와 GR00T의 순위가 실제와 반대인 것도 그대로입니다.
- 실패 양상:
  - GR00T 13번: 못 집음 6, 집은 뒤 못 넣음 7
  - π0 17번: 컵에 닿지도 않음 7, 못 집음 9, 못 넣음 1
- 컵은 논문 사진의 컵처럼 초록 잎 무늬가 있는 흰 종이컵입니다. 우리가 복원한 저장소 예제 영상(PutCupInBowl.mp4)의 컵이 논문의 컵과 같은 물건으로 보입니다.
- 그래서 남은 후보는 물체 배치와 30.5의 시뮬레이터 차이입니다. 지금 가진 자료로는 Cup in Bowl에서 GR00T가 높은 이유를 더 좁힐 수 없습니다.
- 결과는 Drive `results/policy_eval_ours/gr00t/out/cup_in_bowl_ours_real_authbowl/`(GR00T), `results/policy_eval_ours/openpi_local/out/cup_in_bowl_ours_real_authbowl/`(π0)에 있습니다.

### 30.8 Serve Fruits: 같은 수정을 넣은 우리 장면

- 장면 `serve_fruits_ours_real`은 22장의 `serve_fruits_ours`에 30.4와 같은 수정을 넣은 것입니다.
  - 질량: 29장의 방법으로 11단계가 낸 값(배 100 g, 바나나 135 g, 사과 137 g, 주황 접시 36 g, 오렌지 138 g, 청록 쟁반 180 g). 22장은 USD에 들어 있던 값으로, 바나나가 610 g, 쟁반이 30 g이었습니다.
  - 손가락 제한(`ours_serve_fruits_limits`)과 physical 잡기
  - 우리 주황 접시는 그대로 두었습니다(22.4에서 저자 것으로 바꿔 본 물체).
- 25회씩 두 번 돌렸습니다. π0.5와 GR00T N1.6(외부 카메라 하나, 22장과 같음)은 이 머신에서, π0는 super2에서 돌렸습니다.

| 정책 | 25회 ×2 | 합계 | 22장 (assisted, USD 질량) | 논문 시뮬 | 논문 실제 |
|---|---|---|---|---|---|
| π0 | 0, 0 | **0/50 = 0%** | 2/25 = 8% | 4% | 0% |
| π0.5 | 13, 8 | **21/50 = 42%** | 20/25 = 80% | 80% | 72% |
| GR00T N1.6 | 1, 2 | **3/50 = 6%** | 13/25 = 52% | 20% | 4% |

- **순위와 상관은 논문과 맞습니다.** r 0.997(95% 구간 +0.94..+1.00), MMRV 0입니다. 논문은 0.960, 0.016입니다.
- **성공률은 크게 낮아졌습니다.** π0.5는 80%에서 42%로(p = 0.003), GR00T는 52%에서 6%로(p < 0.001) 내려갔습니다. π0.5는 논문 시뮬 80%와 다릅니다(p = 0.003).
  - 22.4에서 GR00T를 높였던 우리 주황 접시가 그대로인데도, GR00T는 논문 실제값(4%)에 가까운 6%가 됐습니다.
  - 질량과 잡기 방식이 함께 바뀌어서, 어느 쪽 때문인지는 이 실행만으로 가를 수 없습니다.
- 실패한 에피소드가 네 단계(바나나와 사과를 집고 각각 접시에 놓기) 중 어디까지 갔는지:
  - π0.5 29번: 두 단계 15, 세 단계 8, 한 단계 4, 없음 2
  - GR00T 47번: 두 단계 21, 한 단계 13, 세 단계 8, 없음 2, 네 단계를 다 거쳤지만 마지막 판정에서 실패 3
  - π0 50번: 없음 32, 두 단계 11, 한 단계 7
- 결과는 Drive `results/policy_eval_ours/openpi_local/out/serve_fruits_ours_real/`(π0·π0.5), `results/policy_eval_ours/gr00t/out/serve_fruits_ours_real/`(GR00T)에 있습니다.

### 30.9 넷째 정책 GR00T N1.7

- 논문 Table G.1의 제로샷 열은 정책 다섯 개(π0, π0.5, GR00T N1.6, GR00T N1.7, DreamZero)입니다. 19장에서는 N1.7의 DROID 체크포인트를 찾지 못해 넷(DreamZero 포함)으로 비교했습니다. 이번에 `nvidia/GR00T-N1.7-DROID`(2026-03 공개)를 찾아 넣었습니다. DreamZero는 46 GB라 12 GB 카드에서는 돌릴 수 없습니다.
- **연결 방법.** 저장소의 평가 단계(`simfoundry/policies/gr00t.py`)는 N1.7을 이미 지원합니다. 그러나 그 클라이언트는 공개되지 않은 저자 서버(`Gr00tG1RealPolicyWrapper`)에 맞춰져 있습니다.
  - 클라이언트는 이미지를 JPEG로, 배열은 npy로 보내고, 배치 차원 없는 동작을 받습니다.
  - 공개 N1.7 서버는 배열을 msgpack_numpy로 주고받고, 그런 래퍼가 없습니다. 그래서 둘은 서로 통하지 않습니다.
  - `repro/policy_eval_ours/gr00t_n17_server.py`가 공개 정책(Isaac-GR00T N1.7 릴리스 `1a1837f`의 `Gr00tPolicy`와 `Gr00tSimPolicyWrapper`)을 클라이언트의 형식으로 내놓습니다. JPEG를 풀고, 동작에서 배치 차원을 뺍니다. 평가 단계는 고치지 않았습니다.
  - `run_local_gr00t.sh`에 `GR00T_VERSION=n17`을 주면 이 서버를 포트 5556에 띄우고, 결과를 `s15_eval_gr00t17_<tag>`로 씁니다. `correlate.py`는 이것을 N1.7 열로 읽습니다.
- **환경.** N1.7은 Python 3.12, torch 2.9.0, flash-attn 2.8.3 공식 휠을 씁니다(`uv sync --frozen`).
  - 저장소에 든 aarch64용 휠이 git-lfs 포인터라서 잠금 파일 없이 풀면 실패합니다.
  - 큰 CUDA 휠은 `UV_HTTP_TIMEOUT=600`으로 받았습니다.
  - 모델(6.9 GB)은 파일마다 curl로 이어받았습니다.
- **메모리와 속도.** 외부 카메라 하나(`nv_franka_droid_single_cam`, N1.7도 외부 카메라 하나와 손목 화면만 읽음)로 서버와 Isaac Sim을 합쳐 10.5 GB입니다. 시뮬레이션은 초당 9–11스텝으로 돕니다.
- **점검.** 1회 점검 영상에서 정책은 컵 위로 가서 자세를 맞추고 집으려 했습니다. 동작이 엉뚱하지 않아 통신과 좌표가 맞는다고 봤습니다.

| 과제 (우리 장면, 모든 수정, 25회 ×2) | GR00T N1.7 | 논문 시뮬 | 논문 실제 |
|---|---|---|---|
| Marker in Cup | 23, 23 → **46/50 = 92%** | 88% | 88% |
| Cup in Bowl | 13, 13 → **26/50 = 52%** | 92% | 92% |

- **Marker in Cup에서는 N1.7이 논문 수준입니다.** 실패 4번 중 3번은 마커를 집은 뒤 컵에 넣지 못했고, 1번은 단계는 다 거쳤지만 마지막 판정에서 실패했습니다. 같은 장면에서 π0.5는 58%였습니다. 그래서 physical 잡기가 강한 정책을 모두 낮추는 것은 아닙니다.
- **Cup in Bowl에서는 논문보다 크게 낮습니다.** 실패 24번은 컵에 닿았지만 못 집음 14번, 집은 뒤 못 넣음 10번입니다. 잡기만 assisted로 바꾼 비교 실행을 super2의 대기열에 넣었습니다.

| 과제 (우리 장면, 모든 수정) | 정책 넷의 r | r 95% 구간 | MMRV | 논문, 같은 넷 | 논문, 다섯 |
|---|---|---|---|---|---|
| Marker in Cup | **0.891** | +0.77..+0.96 | 0.020 | 0.999, 0.010 | 0.995, 0.008 |
| Cup in Bowl | **0.183** | −0.14..+0.53 | 0.170 | 0.889, 0.020 | 0.907, 0.016 |

- Marker in Cup은 정책 넷으로도 논문에 가깝습니다. 어긋남은 π0.5(우리 58%, 실제 92%) 하나입니다.
- Cup in Bowl은 N1.7을 더하자 더 멀어졌습니다. 실제로 92%인 N1.7이 우리 시뮬에서 52%이기 때문입니다.
- Serve Fruits(우리 장면)와 저자 장면 두 과제의 N1.7은 아직 돌고 있습니다.
