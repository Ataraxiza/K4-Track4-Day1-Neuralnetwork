"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations
from sklearn.model_selection import train_test_split
import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_path = f"{processed_dir}/train.npz"
    eval_path = f"{processed_dir}/eval.npz"

    train_data = np.load(train_path)
    eval_data = np.load(eval_path)

    X_train_full = train_data["X"]
    y_train_full = train_data["y"]

    X_eval = eval_data["X"]
    y_eval = eval_data["y"]
    eval_row_id = eval_data["row_id"]

    # Validate train data
    assert X_train_full.dtype == np.float32, (
        f"Expected train X dtype float32, got {X_train_full.dtype}"
    )
    assert y_train_full.dtype == np.int64, (
        f"Expected train y dtype int64, got {y_train_full.dtype}"
    )
    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, (
        f"Expected train X shape (N, 54), got {X_train_full.shape}"
    )
    assert y_train_full.ndim == 1, (
        f"Expected train y shape (N,), got {y_train_full.shape}"
    )
    assert len(X_train_full) == len(y_train_full), (
        "Train X and y have different numbers of samples"
    )

    # Validate eval data
    assert X_eval.dtype == np.float32, (
        f"Expected eval X dtype float32, got {X_eval.dtype}"
    )
    assert y_eval.dtype == np.int64, (
        f"Expected eval y dtype int64, got {y_eval.dtype}"
    )
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, (
        f"Expected eval X shape (N, 54), got {X_eval.shape}"
    )
    assert y_eval.ndim == 1, (
        f"Expected eval y shape (N,), got {y_eval.shape}"
    )
    assert len(X_eval) == len(y_eval), (
        "Eval X and y have different numbers of samples"
    )
    assert len(X_eval) == len(eval_row_id), (
        "Eval X and eval_row_id have different numbers of samples"
    )

    # Check label range.
    assert np.all((y_train_full >= 0) & (y_train_full <= 6)), (
        "Train labels must be in range 0..6"
    )
    assert np.all((y_eval >= 0) & (y_eval <= 6)), (
        "Eval labels must be in range 0..6"
    )

    return (
        X_train_full,
        y_train_full,
        X_eval,
        y_eval,
        eval_row_id,
    )


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(
            f"val_fraction must be between 0 and 1, got {val_fraction}"
        )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
    )

    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(
            f"Expected X_tr shape (N, >= {N_NUMERIC}), got {X_tr.shape}"
        )

    numeric = X_tr[:, :N_NUMERIC]

    # float64 giúp tính thống kê ổn định hơn.
    mean = numeric.mean(axis=0, dtype=np.float64)
    std = numeric.std(axis=0, dtype=np.float64)

    # Các cột có std = 0 không thể standardize theo cách thông thường.
    # Giữ chúng ở mức 0 sau phép biến đổi bằng cách dùng std = 1.
    std = np.where(std == 0.0, 1.0, std)

    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    if X.ndim != 2 or X.shape[1] < N_NUMERIC:
        raise ValueError(
            f"Expected X shape (N, >= {N_NUMERIC}), got {X.shape}"
        )

    mean = np.asarray(mean)
    std = np.asarray(std)

    if mean.shape != (N_NUMERIC,):
        raise ValueError(
            f"Expected mean shape ({N_NUMERIC},), got {mean.shape}"
        )

    if std.shape != (N_NUMERIC,):
        raise ValueError(
            f"Expected std shape ({N_NUMERIC},), got {std.shape}"
        )

    # Không modify X gốc.
    X_out = X.copy()

    X_out[:, :N_NUMERIC] = (
        X_out[:, :N_NUMERIC] - mean
    ) / std

    return X_out.astype(np.float32, copy=False)


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    # 1. Load the predefined train/eval split.
    (
        X_train_full,
        y_train_full,
        X_eval,
        y_eval,
        eval_row_id,
    ) = load_split(processed_dir)

    # 2. Split validation ONLY from training data.
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full,
        y_train_full,
        val_fraction=val_fraction,
        seed=seed,
    )

    # 3. Fit standardizer ONLY on X_tr.
    mean, std = fit_standardizer(X_tr)

    # 4. Apply the SAME train statistics to all three sets.
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    # 5. Move everything to the requested device.
    X_tr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.int64, device=device)

    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.int64, device=device)

    X_eval = torch.tensor(X_eval, dtype=torch.float32, device=device)
    y_eval = torch.tensor(y_eval, dtype=torch.int64, device=device)

    # 6. Print dataset sizes.
    print(f"X_tr   : {tuple(X_tr.shape)}")
    print(f"y_tr   : {tuple(y_tr.shape)}")
    print(f"X_val  : {tuple(X_val.shape)}")
    print(f"y_val  : {tuple(y_val.shape)}")
    print(f"X_eval : {tuple(X_eval.shape)}")
    print(f"y_eval : {tuple(y_eval.shape)}")

    # 7. Majority-class baseline on validation.
    # The majority class is determined from the training set,
    # not from validation or eval.
    classes, counts = torch.unique(y_tr, return_counts=True)
    majority_idx = torch.argmax(counts)
    majority_class = classes[majority_idx]

    majority_pred = torch.full_like(y_val, majority_class)
    majority_acc = (majority_pred == y_val).float().mean().item()

    print(
        f"Majority-class baseline on val: "
        f"{majority_acc:.4f} "
        f"(class {majority_class.item()})"
    )

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "eval_row_id": eval_row_id,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}"
        )

    N = len(X)

    if shuffle:
        perm = torch.randperm(
            N,
            generator=generator,
            device=X.device,
        )
    else:
        perm = torch.arange(
            N,
            device=X.device,
        )

    for i in range(0, N, batch_size):
        idx = perm[i : i + batch_size]
        yield X[idx], y[idx]
