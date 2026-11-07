"""
evaluation.py — PathMNIST A/B/C 실험 공통 평가·로깅 코드 

포함 기능
1. evaluate()            : loss · accuracy · macro F1 + 예측/정답 배열 반환
2. class_report()        : 클래스별 Precision/Recall/F1 + 혼동 행렬
3. Timer                 : GPU 동기화 후 시간 측정 (학습/평가 시간 분리용)
4. EpochLogger           : epoch별 결과를 CSV로 저장
5. BestCheckpoint        : 검증 Macro F1 최고일 때만 체크포인트 저장
6. get_train_eval_loader : 증강 없는 원본 학습 1만 장 로더 (train_eval_idx.npy 사용)

학습 루프 연결 예시 (소유님 train_one_epoch와 합치는 방법)
-------------------------------------------------------------
    logger = EpochLogger(f"log_{exp}_seed{seed}.csv")
    saver  = BestCheckpoint(f"best_{exp}_seed{seed}.pt")

    for epoch in range(1, EPOCHS + 1):
        with Timer(device) as t_train:
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        with Timer(device) as t_eval:
            tr = evaluate(model, train_eval_loader, device, criterion)
            va = evaluate(model, val_loader, device, criterion)

        logger.log(epoch, t_train.elapsed, train_loss, tr, va, t_eval.elapsed)
        saver.update(model, epoch, va["macro_f1"])

    # 학습 종료 후: saver.load(model) → 테스트 1회 평가 (모델·설정 확정 후에만!)
"""

import csv
import os
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

CLASS_NAMES = [
    "Adipose", "Background", "Debris", "Lymphocytes", "Mucus",
    "Smooth muscle", "Normal colon mucosa", "Cancer-assoc. stroma", "Adenocarcinoma epi.",
]
NUM_CLASSES = 9


# ---------------------------------------------------------------
# 1. 평가
# ---------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """loader 전체를 평가해 dict를 반환한다.

    반환: {"loss", "acc", "macro_f1", "y_true", "y_pred"}
    - 라벨은 MedMNIST 기본 형태 [N, 1]도 자동으로 [N]으로 맞춘다.
    - criterion을 주지 않으면 CrossEntropyLoss를 사용한다.
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    was_training = model.training
    model.eval()

    total_loss, n = 0.0, 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.view(-1).long().to(device, non_blocking=True)

        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        n += labels.size(0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(outputs.argmax(dim=1).cpu().numpy())

    if was_training:
        model.train()

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return {
        "loss": total_loss / n,
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(NUM_CLASSES)), zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def class_report(y_true, y_pred):
    """클래스별 Precision/Recall/F1/개수와 혼동 행렬을 반환한다."""
    labels = list(range(NUM_CLASSES))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = [
        {"class": i, "name": CLASS_NAMES[i], "precision": float(p[i]),
         "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
        for i in labels
    ]
    return rows, cm


def print_class_report(y_true, y_pred):
    rows, _ = class_report(y_true, y_pred)
    print(f"{'ID':<3}{'Class':<22}{'Prec':>7}{'Rec':>7}{'F1':>7}{'N':>7}")
    for r in rows:
        print(f"{r['class']:<3}{r['name']:<22}{r['precision']:>7.4f}"
              f"{r['recall']:>7.4f}{r['f1']:>7.4f}{r['support']:>7}")


# ---------------------------------------------------------------
# 2. 시간 측정
# ---------------------------------------------------------------
class Timer:
    """GPU 연산이 끝날 때까지 기다린 뒤 시간을 잰다.

    with Timer(device) as t:
        ...
    print(t.elapsed)  # 초
    """

    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.elapsed = 0.0

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def __enter__(self):
        self._sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.elapsed = time.perf_counter() - self._start
        return False


# ---------------------------------------------------------------
# 3. epoch별 로그 (CSV)
# ---------------------------------------------------------------
LOG_COLUMNS = [
    "epoch", "train_time_sec", "train_loss_running",
    "train_eval_loss", "train_eval_acc", "train_eval_f1",
    "val_loss", "val_acc", "val_f1", "eval_time_sec",
]


class EpochLogger:
    """epoch마다 한 줄씩 CSV에 기록한다. 같은 파일명이 있으면 새로 덮어쓴다."""

    def __init__(self, path):
        self.path = path
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

    def log(self, epoch, train_time, train_loss, train_eval, val, eval_time, verbose=True):
        row = [
            epoch, round(train_time, 3), round(train_loss, 5),
            round(train_eval["loss"], 5), round(train_eval["acc"], 5), round(train_eval["macro_f1"], 5),
            round(val["loss"], 5), round(val["acc"], 5), round(val["macro_f1"], 5),
            round(eval_time, 3),
        ]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        if verbose:
            print(f"[{epoch:>3}] train {train_time:6.1f}s loss {train_loss:.4f} | "
                  f"train1만 F1 {train_eval['macro_f1']:.4f} | "
                  f"val loss {val['loss']:.4f} acc {val['acc']:.4f} F1 {val['macro_f1']:.4f} | "
                  f"eval {eval_time:5.1f}s")


# ---------------------------------------------------------------
# 4. 최고 검증 Macro F1 체크포인트
# ---------------------------------------------------------------
class BestCheckpoint:
    """검증 Macro F1이 이전 최고보다 높을 때만 저장한다."""

    def __init__(self, path):
        self.path = path
        self.best_f1 = -1.0
        self.best_epoch = None

    def update(self, model, epoch, val_f1):
        if val_f1 > self.best_f1:
            self.best_f1, self.best_epoch = val_f1, epoch
            torch.save({"epoch": epoch, "val_macro_f1": val_f1,
                        "model_state": model.state_dict()}, self.path)
            return True
        return False

    def load(self, model, device="cpu"):
        ckpt = torch.load(self.path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        return ckpt["epoch"], ckpt["val_macro_f1"]


# ---------------------------------------------------------------
# 5. 원본 학습 1만 장 로더
# ---------------------------------------------------------------
def get_train_eval_loader(idx_path="train_eval_idx.npy", batch_size=256, num_workers=2):
    """증강 없는(A 변환) 학습 이미지 중 고정된 1만 장을 반환한다.
    A/B/C 모두 같은 파일을 써야 공정한 비교가 된다."""
    from medmnist import PathMNIST
    from torch.utils.data import DataLoader, Subset
    from pathmnist_transforms import get_transform

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"{idx_path}가 없습니다. 노션에서 받아 업로드하세요.")

    idx = np.load(idx_path)
    full = PathMNIST(split="train", download=True, transform=get_transform("A"))
    return DataLoader(Subset(full, idx.tolist()), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)


def param_count(model):
    """모델 파라미터 수 (A/B/C가 같아야 정상)."""
    return sum(p.numel() for p in model.parameters())
