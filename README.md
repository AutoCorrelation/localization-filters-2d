# 코드 실행 방법  
- git clone 후 폴더 디렉토리에 `/data` 폴더 추가  
- src/`main.m` 파일 실행  
- 실행 시 do preSimulate? Y/N:  라고 묻는데 `/data` 내 시뮬레이션 데이터가 있으면 N, 초기 실행이라면 Y

 # parameter  
 |x|number of x|
 |--|--|
 |Particles|1e3|
 |Iteration(KF)|1e3|
 |Iteration(PF)|1e3|  

# results  
 |x|0.01|0.1|1|10|100|
 |--|--|-|-|-|-|
 |KF|0.0897|0.2815|0.8853|2.6917|8.1255| 
 |KF1|0.0852|0.2671|0.8487|2.6306|7.5216|
 |PF|0.0905|0.2840|0.8922|2.7027|8.0361| 
 |PF(1e4)|0.0834|0.2667|0.8504|2.6644|7.4672|

---

# 머신러닝 / PyTorch 활용

## 📊 데이터 흐름

### MATLAB에서 HDF5 형식으로 데이터 저장

```
MATLAB (src/Env.m)
├─ preSimulate()      → MAT/CSV 파일 저장 (기본 방식)
└─ preSimulateH5()    → HDF5 파일 저장 ✓ (권장)
       ↓
  data/simulation_data.h5
    ├─ /z              (6, iterations, points, noises)     → 상대거리 측정값 (TDOA)
    ├─ /toaPos         (2, iterations, points, noises)     → 추정된 위치 (노이즈 포함)
    ├─ /realPos        (2, iterations, points)             → 실제 위치 (지상진실) ✓ NEW
    ├─ /R              (6, 6, iterations, points, noises)  → 측정 공분산
    ├─ /Q              (2, 2, noises)                      → 프로세스 노이즈 공분산
    ├─ /P0             (2, 2, noises)                      → 초기 상태 공분산
    ├─ /processNoise   (2, iterations, noises)            → 프로세스 노이즈
    ├─ /toaNoise       (2, iterations, noises)            → 측정 노이즈
    └─ /processbias    (2, noises)                        → 프로세스 바이어스
```

### MATLAB 데이터 생성 (HDF5 형식)

**Step 1: MATLAB에서 실행**

```matlab
% src/main.m 대신 다음 명령어 실행
env = Env(1000);        % 1000개 반복
env.preSimulateH5();    % HDF5 형식으로 저장
% 결과: data/simulation_data.h5 생성
```

**주요 파라미터 (src/Env.m)**
- `numIterations`: 시뮬레이션 반복 횟수
- `numPoints`: 테스트 위치 개수 (기본값: 10)
- `Anchor`: 4개 기지국 위치 좌표
- `noiseVariance`: 5가지 노이즈 레벨 (0.01, 0.1, 1, 10, 100)

---

## 🐍 PyTorch 머신러닝 모델

### 폴더 구조
```
pytorch/
├── load_h5_data.py                          # HDF5 데이터 로드 유틸리티
├── dataset.py                               # PyTorch Dataset 클래스
├── 01_data_loading_tutorial.ipynb          # 데이터 로드 및 기본 DNN 모델 튜토리얼
└── 02_transformer_position_predictor.ipynb # 트랜스포머 모델 (시계열 예측)
```

### 필요 라이브러리
```bash
pip install torch torchvision torchaudio
pip install h5py numpy matplotlib
pip install jupyter
```

---

## 📓 주피터 노트북 가이드

### 1️⃣ 01_data_loading_tutorial.ipynb
기본 데이터 로드 및 간단한 신경망(DNN) 모델

**주요 내용:**
- HDF5 파일 구조 확인
- PyTorch DataLoader 설정
- 데이터 정규화 및 배치 처리
- 간단한 위치 예측 모델 (z → toaPos)
- 훈련/검증 루프 및 손실 함수 시각화

**사용 시나리오:**
- 데이터 구조 이해
- 기본 신경망 훈련
- 예측 성능 평가 (MAE, RMSE)

**예상 실행 시간:** ~10-15분

---

### 2️⃣ 02_transformer_position_predictor.ipynb
시계열 위치 데이터를 이용한 트랜스포머 모델

**주요 특징:**
- **입력**: 이전 10개 시점의 위치 시계열
- **출력**: 다음 시점의 예측 위치
- **모델**: Multi-head Self-Attention Transformer Encoder
- **학습 목표**: 실제 위치 (**realPos**/지상진실) ✓ **🔑 중요**

**학습 데이터:**
- **입력 시계열**: 노이즈가 있는 위치 (toaPos)
- **목표값 (Target)**: 실제 정확한 위치 (realPos)
- 즉, 모델이 **노이즈 제거 및 위치 예측**을 동시에 학습합니다!

**모델 구조:**
```
Input Sequence (seq_len=10, dim=2)
    ↓
Input Projection (2 → 64)
    ↓
Positional Encoding
    ↓
Transformer Encoder (2 layers, 4 heads)
    ├─ Self-Attention
    ├─ Feed-Forward (64 → 256 → 64)
    └─ Layer Normalization
    ↓
Output Projection (64 → 2)
    ↓
Predicted Next Position (dim=2)
```

**주요 특징:**
- **Positional Encoding**: 시계열 순서 정보 인코딩
- **Multi-head Attention**: 4개 헤드로 다양한 특징 학습
- **시계열 처리**: 위치 궤적의 시간적 패턴 학습

**예상 성능:**
- 노이즈 낮음 (0.01~0.1): RMSE < 0.15
- 중간 노이즈 (1): RMSE < 1.0
- 높은 노이즈 (10~100): RMSE 증가

---

## 🚀 빠른 시작 가이드

### 1단계: MATLAB에서 데이터 생성
```matlab
cd src
env = Env(1000);
env.preSimulateH5();
% data/simulation_data.h5 파일 생성됨
```

### 2단계: Python 환경 설정
```bash
cd pytorch
# 주피터 시작
jupyter notebook
```

### 3단계: 노트북 실행
- `01_data_loading_tutorial.ipynb` 먼저 실행 (데이터 이해)
- `02_transformer_position_predictor.ipynb` 실행 (트랜스포머 모델)

### 4단계: 모델 저장 및 사용
```python
# 모델 저장 (자동)
torch.save({
    'model_state_dict': model.state_dict(),
    'best_val_loss': best_val_loss,
}, 'transformer_position_predictor.pth')

# 모델 로드
checkpoint = torch.load('transformer_position_predictor.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 📈 데이터 특성

### 노이즈 레벨별 위치 분포
| 노이즈 | 특징 | 추천 모델 |
|--------|------|----------|
| 0.01 | 매우 정확 | 모든 모델 |
| 0.1 | 정확 | DNN, Transformer |
| 1 | 중간 | Transformer ⭐ |
| 10 | 낮은 정확도 | Transformer + 정규화 |
| 100 | 매우 노이즈 많음 | 앙상블 모델 |

**트랜스포머의 장점:**
- 시계열의 시간적 의존성 학습
- 노이즈에 대한 강건성
- Attention 가중치로 해석 가능성 제공

---

## 🔧 커스터마이징 옵션

### 모델 하이퍼파라미터 조정 (02_transformer_position_predictor.ipynb)

```python
# 트랜스포머 설정 변경
model = TransformerPositionPredictor(
    input_dim=2,              # 입력 차원 (x, y)
    d_model=64,               # 트랜스포머 임베딩 차원
    nhead=4,                  # Attention 헤드 수
    num_layers=2,             # Transformer 레이어 수
    dim_feedforward=256,      # 피드포워드 숨겨진 차원
    dropout=0.1,              # Dropout 비율
    output_dim=2              # 출력 차원 (x, y)
)

# 훈련 파라미터
num_epochs = 30             # 에포크 수
batch_size = 32             # 배치 크기
seq_len = 10                # 입력 시계열 길이
early_stopping_patience = 5 # 조기 종료 대기 에포크
```

---

## 📊 결과 해석

### 트랜스포머 모델 성능 메트릭

1. **RMSE (Root Mean Square Error)**
   - 모든 오류를 동등하게 취급
   - 큰 오류에 더 많은 가중치

2. **MAE (Mean Absolute Error)**
   - 평균 절대 오류
   - 해석하기 쉬운 메트릭

3. **Mean Distance Error**
   - 실제 위치와 예측 위치 간의 거리
   - 물리적 의미 있는 메트릭

### Attention 가중치 분석
트랜스포머의 Attention 가중치를 시각화하면:
- 어떤 과거 시점이 가장 영향력 있는지 확인
- 모델의 의사결정 프로세스 이해
- 물리적으로 타당한 패턴인지 검증

---

## 📝 추천용 다음 단계

1. **노이즈 레벨별 모델 분리**
   - 각 노이즈 레벨마다 별도 모델 훈련
   - 성능 향상 가능

2. **Multi-step Forecasting**
   - 여러 스텝 단위로 예측
   - 재귀적 예측 구현

3. **Attention 시각화**
   - Attention 히트맵 생성
   - 모델 해석가능성 증대

4. **하이브리드 모델**
   - 칼만 필터 + 트랜스포머
   - 물리 제약 조건 적용

5. **Ensemble Learning**
   - 여러 모델 조합
   - 예측 견고성 향상

---

## ⚠️ 트러블슈팅

### "HDF5 파일을 찾을 수 없다" 에러
```
해결방법:
1. MATLAB에서 env.preSimulateH5() 실행했는지 확인
2. data/simulation_data.h5 파일 존재 확인
3. 파일 경로 확인 (상대경로 '../data/simulation_data.h5')
```

### GPU 메모리 부족
```
해결방법:
1. batch_size 감소 (32 → 16 또는 8)
2. seq_len 감소 (10 → 5)
3. num_layers 감소 (2 → 1)
4. CPU에서 직접 실행 (device = 'cpu')
```

### 훈련이 진행되지 않음 (손실 값 변화 없음)
```
해결방법:
1. Learning rate 조정 (1e-3 → 1e-2 또는 1e-4)
2. 배치 크기 증가
3. 데이터 정규화 확인
4. 조기 종료 patience 값 증가
```

---

## 📚 참고 자료

- [PyTorch Transformer 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html)
- [Attention is All You Need (원본 논문)](https://arxiv.org/abs/1706.03762)
- [Time Series Forecasting with Transformers](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

---

## 👥 기여 및 문의

질문이나 개선 사항이 있으면 이슈를 제출해주세요.


 
 
