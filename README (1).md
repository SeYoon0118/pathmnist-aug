# PathMNIST 데이터 증강 실험 (심층학습 1조)

데이터 증강이 CNN 기반 병리 이미지 분류의 일반화 성능과 학습 비용에 미치는 영향을 비교합니다.
같은 CNN·같은 데이터 분할에서 학습 데이터 처리 방식만 바꿉니다.

| 조건 | 학습 데이터 처리 |
|---|---|
| A | 원본 |
| B | 90° 단위 무작위 회전 + 50% 좌우 반전 |
| C | B + 색감 변화(ColorJitter 0.1/0.1/0.1, hue 0.01, p=0.8) |

## 파일 구성
| 파일 | 내용 | 담당 |
|---|---|---|
| `pathmnist_transforms.py` | A/B/C 변환, 데이터셋 불러오기 | 강윤서 |
| `model.py` | CNN 모델 | 박소유 |
| `train.py` | 학습 루프, 시드 고정 | 박소유 |
| `evaluation.py` | 평가 지표, 시간 측정, 로그, 체크포인트 | 이세윤 |
| `train_eval_idx.npy` | 과적합 확인용 원본 학습 1만 장 인덱스 (seed 42, 층화 추출) | 이세윤 |

## Colab에서 사용하기
```python
!pip install -q medmnist
!git clone https://github.com/<계정>/pathmnist-aug.git
%cd pathmnist-aug
!git pull   # 다시 실행할 때 최신 코드 받기

from pathmnist_transforms import get_datasets
from evaluation import *
```

결과(로그 CSV, 그래프, 체크포인트)는 구글 드라이브 공유 폴더에 저장합니다.
```python
from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = "/content/drive/MyDrive/<공유폴더>/results"
```

## 협업 규칙
1. 공용 코드는 이 저장소에만 둡니다. 사본을 고쳐서 실험하지 않습니다.
2. 코드를 바꾸면 단톡에 한 줄 공지합니다. (예: "transforms에 정규화 추가")
3. 결과 파일 이름에 조건·시드·날짜를 넣습니다. (예: `log_C_seed1_1110.csv`)
4. 테스트셋은 모델·설정을 확정한 뒤 조건별로 한 번만 평가합니다.

## 확정 대기 중
- 입력 정규화 사용 여부
- 시드 3개, epoch 수
