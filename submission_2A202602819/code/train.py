"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Giúp kết quả reproducible hơn trên CUDA.
    # Có thể làm chậm một chút.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm)

    if cm.shape != (7, 7):
        raise ValueError(
            f"Expected confusion matrix shape (7, 7), got {cm.shape}"
        )

    tp = np.diag(cm).astype(np.float64)

    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) > 0,
    )

    recall = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp),
        where=(tp + fn) > 0,
    )

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    model.eval()

    predictions = []

    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size]

        logits = model(xb)
        pred = torch.argmax(logits, dim=1)

        predictions.append(pred.to(dtype=torch.int64))

    if not predictions:
        return torch.empty(
            0,
            dtype=torch.int64,
            device=X.device,
        )

    return torch.cat(predictions, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same length, got {len(X)} and {len(y)}"
        )

    if len(X) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Giữ confusion matrix ở CPU để tránh tạo tensor lớn không cần thiết.
    cm = np.zeros((7, 7), dtype=np.int64)

    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size]
        yb = y[i : i + batch_size]

        logits = model(xb)

        # Evaluation luôn dùng fp32 để metric/loss ổn định.
        if loss_name.lower() == "ce":
            batch_loss = F.cross_entropy(
                logits,
                yb,
                reduction="sum",
            )

        elif loss_name.lower() == "mse":
            one_hot = F.one_hot(
                yb,
                num_classes=7,
            ).to(dtype=logits.dtype)

            batch_loss = F.mse_loss(
                logits,
                one_hot,
                reduction="sum",
            )

        else:
            raise ValueError(
                f"Unknown loss '{loss_name}'. Expected 'ce' or 'mse'."
            )

        pred = torch.argmax(logits, dim=1)

        total_loss += float(batch_loss.item())

        total_correct += int(
            (pred == yb).sum().item()
        )

        batch_n = int(yb.numel())
        total_samples += batch_n

        # Chuyển batch nhỏ về CPU/numpy để cập nhật confusion matrix.
        true_np = yb.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        np.add.at(cm, (true_np, pred_np), 1)

    loss = total_loss / total_samples
    acc = total_correct / total_samples
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": float(loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
    }


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    loss_name = loss_name.lower()

    if loss_name == "ce":
        return F.cross_entropy(
            logits,
            y,
            reduction="mean",
        )

    if loss_name == "mse":
        one_hot = F.one_hot(
            y,
            num_classes=7,
        ).to(dtype=logits.dtype)

        return F.mse_loss(
            logits,
            one_hot,
            reduction="mean",
        )

    raise ValueError(
        f"Unknown loss '{loss_name}'. Expected 'ce' or 'mse'."
    )

# ---------------------------------------------------------------------
# Helpers for mixed precision
# ---------------------------------------------------------------------

def _get_device(model):
    """Lấy device từ parameter đầu tiên của model."""
    return next(model.parameters()).device


def _autocast_context(precision: str, device: torch.device):
    """Trả về context manager cho forward/loss."""
    precision = precision.lower()

    if precision == "fp32":
        return torch.autocast(
            device_type=device.type,
            enabled=False,
        )

    if precision == "fp16":
        # FP16 autocast chủ yếu dùng trên CUDA.
        if device.type != "cuda":
            raise ValueError(
                "precision='fp16' requires a CUDA device."
            )

        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )

    if precision == "bf16":
        # BF16 được hỗ trợ tốt trên CUDA mới và một số CPU.
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=True,
        )

    raise ValueError(
        f"Unknown precision '{precision}'. "
        "Expected 'fp32', 'fp16' or 'bf16'."
    )


def _make_grad_scaler(precision: str, device: torch.device):
    """Tạo GradScaler chỉ cho FP16."""
    if precision != "fp16":
        return None

    if device.type != "cuda":
        raise ValueError(
            "GradScaler for precision='fp16' requires CUDA."
        )

    # Tương thích PyTorch 2.x.
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
    except (AttributeError, TypeError):
        # Fallback cho một số phiên bản PyTorch cũ hơn.
        return torch.cuda.amp.GradScaler(
            enabled=True,
        )


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    required_data_keys = [
        "X_tr",
        "y_tr",
        "X_val",
        "y_val",
    ]

    for key in required_data_keys:
        if key not in data:
            raise KeyError(f"Missing data key: {key}")

    if cfg.get("lr") is None:
        raise ValueError(
            "cfg['lr'] must be set before run_experiment(). "
            "Choose it using validation first."
        )

    loss_name = str(cfg.get("loss", "ce")).lower()
    optimizer_name = str(
        cfg.get("optimizer", "sgd_momentum")
    ).lower()
    precision = str(
        cfg.get("precision", "fp32")
    ).lower()

    if loss_name not in ("ce", "mse"):
        raise ValueError(
            f"Unknown loss '{loss_name}'."
        )

    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(
            f"Unknown precision '{precision}'."
        )

    if cfg["epochs"] <= 0:
        raise ValueError("epochs must be > 0")

    if cfg["batch"] <= 0:
        raise ValueError("batch must be > 0")

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------

    set_seed(cfg["seed"])

    X_tr = data["X_tr"]
    y_tr = data["y_tr"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    device = X_tr.device

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    hidden = tuple(cfg["hidden"])

    if hidden not in EXPECTED_PARAMS:
        raise ValueError(
            f"Architecture {hidden} is not listed in EXPECTED_PARAMS."
        )

    model = MLP(
        hidden=hidden,
        dropout=cfg["dropout"],
        init=cfg["init"],
    )

    actual_params = count_params(model)
    expected_params = EXPECTED_PARAMS[hidden]

    assert actual_params == expected_params, (
        f"Parameter count mismatch for hidden={hidden}: "
        f"got {actual_params}, expected {expected_params}"
    )

    model = model.to(device)

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------

    optimizer = build_optimizer(
        optimizer_name,
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        momentum=float(cfg.get("momentum", 0.9)),
    )

    # ---------------------------------------------------------------
    # Mixed precision
    # ---------------------------------------------------------------

    scaler = _make_grad_scaler(
        precision,
        device,
    )

    # ---------------------------------------------------------------
    # Fixed generator for batch shuffling
    # ---------------------------------------------------------------

    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg["seed"]))

    # ---------------------------------------------------------------
    # Initial loss before any parameter update
    # ---------------------------------------------------------------

    step0_metrics = evaluate(
        model,
        X_val,
        y_val,
        loss_name=loss_name,
    )

    step0_loss = step0_metrics["loss"]

    print(
        f"[{cfg.get('exp_id', 'experiment')}] "
        f"step0 val_loss={step0_loss:.6f}"
    )

    # ---------------------------------------------------------------
    # History
    # ---------------------------------------------------------------

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = None
    best_state = None

    diverged = False

    # ---------------------------------------------------------------
    # Optional GPU memory measurement
    # ---------------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_start = time.perf_counter()

        model.train()

        epoch_grad_norms = []

        # New deterministic generator state is consumed every epoch.
        batch_iterator = iterate_batches(
            X_tr,
            y_tr,
            cfg["batch"],
            generator=generator,
            shuffle=True,
        )

        for xb, yb in batch_iterator:

            optimizer.zero_grad(set_to_none=True)

            # -------------------------------------------------------
            # Forward + loss only inside autocast.
            # -------------------------------------------------------
            with _autocast_context(
                precision,
                device,
            ):
                logits = model(xb)
                loss = compute_loss(
                    logits,
                    yb,
                    loss_name,
                )

            # -------------------------------------------------------
            # Detect bad loss BEFORE backward.
            # -------------------------------------------------------
            if not torch.isfinite(loss).item():
                diverged = True

                print(
                    f"[{cfg.get('exp_id', 'experiment')}] "
                    f"DIVERGED at epoch={epoch}: "
                    f"loss={loss.item()}"
                )

                break

            # -------------------------------------------------------
            # Backward
            # -------------------------------------------------------
            if scaler is not None:
                scaler.scale(loss).backward()

                # Gradients are still scaled here.
                # Unscale BEFORE clipping.
                if cfg.get("clip_norm") is not None:
                    scaler.unscale_(optimizer)

            else:
                loss.backward()

            # -------------------------------------------------------
            # Gradient norm BEFORE clipping
            # -------------------------------------------------------
            try:
                gn = clip_gradients(
                    model.parameters(),
                    cfg.get("clip_norm"),
                )
            except RuntimeError as exc:
                diverged = True

                print(
                    f"[{cfg.get('exp_id', 'experiment')}] "
                    f"DIVERGED during gradient clipping at "
                    f"epoch={epoch}: {exc}"
                )

                break

            if not np.isfinite(gn):
                diverged = True

                print(
                    f"[{cfg.get('exp_id', 'experiment')}] "
                    f"DIVERGED at epoch={epoch}: "
                    f"gradient norm={gn}"
                )

                break

            epoch_grad_norms.append(gn)

            # -------------------------------------------------------
            # Optimizer update
            # -------------------------------------------------------
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # -----------------------------------------------------------
        # Stop immediately if diverged.
        # -----------------------------------------------------------
        if diverged:
            break

        # -----------------------------------------------------------
        # Evaluation at end of epoch
        # -----------------------------------------------------------
        train_metrics = evaluate(
            model,
            X_tr,
            y_tr,
            loss_name=loss_name,
        )

        val_metrics = evaluate(
            model,
            X_val,
            y_val,
            loss_name=loss_name,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_time = time.perf_counter() - epoch_start

        mean_grad_norm = (
            float(np.mean(epoch_grad_norms))
            if epoch_grad_norms
            else float("nan")
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(mean_grad_norm)
        history["epoch_time_s"].append(epoch_time)

        # -----------------------------------------------------------
        # Best model = lowest validation loss
        # -----------------------------------------------------------
        if np.isfinite(val_metrics["loss"]):
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch

                # Deep-copy to CPU so later optimizer updates cannot
                # mutate the stored best state.
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                }

        print(
            f"[{cfg.get('exp_id', 'experiment')}] "
            f"epoch {epoch:02d}/{cfg['epochs']} | "
            f"train_loss={train_metrics['loss']:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"grad_norm={mean_grad_norm:.4f} | "
            f"time={epoch_time:.2f}s"
        )

    # ---------------------------------------------------------------
    # Handle experiment that diverged before producing best state.
    # ---------------------------------------------------------------

    if best_state is None:
        # Save current state only as a diagnostic fallback.
        # This experiment is marked diverged and must not be selected
        # as the best LR.
        best_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.state_dict().items()
        }

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    if history["epoch"]:
        final_train_loss = float(history["train_loss"][-1])
        final_val_loss = float(history["val_loss"][-1])

    else:
        final_train_loss = float("nan")
        final_val_loss = float("nan")

    if best_epoch is not None:
        best_idx = history["epoch"].index(best_epoch)

        best_val_acc = float(
            history["val_acc"][best_idx]
        )

        best_val_macro_f1 = float(
            history["val_macro_f1"][best_idx]
        )

    else:
        best_val_acc = float("nan")
        best_val_macro_f1 = float("nan")

    if history["epoch_time_s"]:
        time_per_epoch_s = float(
            np.mean(history["epoch_time_s"])
        )
    else:
        time_per_epoch_s = float("nan")

    if device.type == "cuda":
        peak_mem_MB = float(
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )
    else:
        peak_mem_MB = 0.0

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "val_acc": best_val_acc,
        "val_macro_f1": best_val_macro_f1,
        "time_per_epoch_s": time_per_epoch_s,
        "peak_mem_MB": peak_mem_MB,
        "diverged": bool(diverged),
    }

    # ---------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------

    return {
        "cfg": dict(cfg),
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)

    if row_id.ndim != 1:
        raise ValueError(
            f"row_id must be 1-D, got shape {row_id.shape}"
        )

    if preds.ndim != 1:
        raise ValueError(
            f"preds must be 1-D, got shape {preds.shape}"
        )

    if len(row_id) != len(preds):
        raise ValueError(
            f"row_id and preds have different lengths: "
            f"{len(row_id)} vs {len(preds)}"
        )

    if len(row_id) == 0:
        raise ValueError("Cannot write an empty prediction file.")

    # Kiểm tra integer.
    if not np.all(row_id == row_id.astype(np.int64)):
        raise ValueError("row_id must contain integers.")

    if not np.all(preds == preds.astype(np.int64)):
        raise ValueError("pred must contain integers.")

    row_id = row_id.astype(np.int64)
    preds = preds.astype(np.int64)

    # Kiểm tra nhãn.
    if np.any((preds < 0) | (preds > 6)):
        raise ValueError(
            "pred must be in range 0..6."
        )

    # Kiểm tra duplicate row_id.
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError(
            "row_id contains duplicates."
        )

    # Tạo parent directory nếu cần.
    parent = os.path.dirname(os.path.abspath(path))

    if parent:
        os.makedirs(parent, exist_ok=True)

    # Ghi CSV chính xác hai cột.
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(["row_id", "pred"])

        for rid, pred in zip(row_id, preds):
            writer.writerow([
                int(rid),
                int(pred),
            ])

    print(
        f"Wrote {len(preds)} predictions to {path}"
    )


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    if "best_state" not in result:
        raise KeyError(
            "result does not contain 'best_state'."
        )

    required_keys = [
        "X_eval",
        "eval_row_id",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Missing data key: {key}"
            )

    # ---------------------------------------------------------------
    # Determine device from X_eval.
    # ---------------------------------------------------------------
    X_eval = data["X_eval"]
    device = X_eval.device

    # ---------------------------------------------------------------
    # Re-create exactly the same architecture.
    # ---------------------------------------------------------------
    hidden = tuple(cfg["hidden"])

    if hidden not in EXPECTED_PARAMS:
        raise ValueError(
            f"Architecture {hidden} is not listed in EXPECTED_PARAMS."
        )

    model = MLP(
        hidden=hidden,
        dropout=cfg["dropout"],
        init=cfg["init"],
    )

    actual_params = count_params(model)
    expected_params = EXPECTED_PARAMS[hidden]

    assert actual_params == expected_params, (
        f"Parameter count mismatch: "
        f"got {actual_params}, expected {expected_params}"
    )

    # best_state is stored on CPU.
    model.load_state_dict(
        result["best_state"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------------
    # Predict eval.
    # ---------------------------------------------------------------
    preds = predict(
        model,
        X_eval,
    )

    preds_np = preds.detach().cpu().numpy()

    # ---------------------------------------------------------------
    # Write submission.
    # ---------------------------------------------------------------
    write_predictions(
        data["eval_row_id"],
        preds_np,
        pred_path,
    )

    print(
        "\nFinal evaluation predictions created."
    )
    print(
        f"Configuration: {cfg.get('exp_id', 'final')}"
    )
    print(
        f"Prediction file: {pred_path}"
    )
    print(
        f"Number of predictions: {len(preds_np)}"
    )
