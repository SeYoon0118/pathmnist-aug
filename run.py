#실험함수 정의
import os, torch
from torch.utils.data import DataLoader, Subset
from model import PathMNISTCNN
from train import set_seed, seed_worker, train_one_epoch
from evaluation import evaluate, Timer, EpochLogger, BestCheckpoint, param_count

def run_experiment(exp, seed, epochs, train_ds, val_ds, train_eval_ds,
                   save_dir="results", batch_size=128, lr=1e-3, num_workers=2, subset=None):
    """한 조건(exp)·한 시드로 학습하고 epoch별 로그와 최고 체크포인트를 저장한다."""
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if subset:  # 시험 실행용: 앞쪽 일부만 사용
        train_ds = Subset(train_ds, range(subset))

    set_seed(seed)                       # 1. 시드 고정 → 초기 가중치가 A/B/C에서 같아짐
    model = PathMNISTCNN().to(device)

    g = torch.Generator().manual_seed(seed)   # 2. 섞는 순서·증강 난수도 고정
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=256, shuffle=False, num_workers=num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tag = f"{exp}_seed{seed}"
    logger = EpochLogger(os.path.join(save_dir, f"log_{tag}.csv"))
    saver = BestCheckpoint(os.path.join(save_dir, f"best_{tag}.pt"))

    print(f"=== {tag} | device={device} | params={param_count(model):,} | train={len(train_ds):,}장 ===")
    for epoch in range(1, epochs + 1):
        with Timer(device) as t_train:
            train_loss = train_one_epoch(model=model, train_loader=train_loader,
                                         criterion=criterion, optimizer=optimizer, device=device)
        with Timer(device) as t_eval:
            tr = evaluate(model, train_eval_loader, device, criterion)
            va = evaluate(model, val_loader, device, criterion)
        logger.log(epoch, t_train.elapsed, train_loss, tr, va, t_eval.elapsed)
        saver.update(model, epoch, va["macro_f1"])

    print(f"최고 검증 Macro F1 {saver.best_f1:.4f} (epoch {saver.best_epoch}) → {saver.path}")
    return model, saver