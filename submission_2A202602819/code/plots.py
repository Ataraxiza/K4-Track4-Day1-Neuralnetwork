"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import os

def _get_config_text(result: dict) -> str:
    """Create a compact description of the experiment configuration."""

    exp_id = result.get("exp_id", "unknown")

    # Configuration may be stored either directly in result or
    # inside result["config"].
    config = result.get("config", {})

    if not isinstance(config, dict):
        config = {}

    parts = [f"exp_id={exp_id}"]

    keys = [
        "optimizer",
        "lr",
        "batch_size",
        "epochs",
        "hidden",
        "dropout",
        "weight_decay",
    ]

    for key in keys:
        value = config.get(key, result.get(key))

        if value is not None:
            parts.append(f"{key}={value}")

    return ", ".join(parts)


def _get_history(result: dict, key: str):
    """Get a history array from result.

    Supports both:
        result["history"][key]
    and:
        result[key]
    """

    history = result.get("history")

    if isinstance(history, dict) and key in history:
        return history[key]

    return result.get(key, [])


def _mark_best_epoch(ax, result: dict):
    """Mark best_epoch on an axis if available."""

    best_epoch = result.get("best_epoch")

    if best_epoch is None:
        return

    # Epoch numbering may be 0-based or 1-based depending on train.py.
    ax.axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"best epoch={best_epoch}",
    )

def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    # ------------------------------------------------------------
    # Read history
    # ------------------------------------------------------------

    train_loss = _get_history(result, "train_loss")
    val_loss = _get_history(result, "val_loss")
    val_acc = _get_history(result, "val_acc")
    val_macro_f1 = _get_history(result, "val_macro_f1")
    grad_norm = _get_history(result, "grad_norm")

    # Determine number of epochs.
    histories = [
        train_loss,
        val_loss,
        val_acc,
        val_macro_f1,
        grad_norm,
    ]

    n_epochs = max(
        (len(h) for h in histories if h is not None),
        default=0,
    )

    if n_epochs == 0:
        raise ValueError(
            "No epoch history found in result."
        )

    epochs = range(1, n_epochs + 1)

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    # ------------------------------------------------------------
    # Panel 1: Loss
    # ------------------------------------------------------------

    ax = axes[0]

    if train_loss:
        ax.plot(
            range(1, len(train_loss) + 1),
            train_loss,
            label="train_loss",
            linewidth=2,
        )

    if val_loss:
        ax.plot(
            range(1, len(val_loss) + 1),
            val_loss,
            label="val_loss",
            linewidth=2,
        )

    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    _mark_best_epoch(ax, result)

    # ------------------------------------------------------------
    # Panel 2: Validation metrics
    # ------------------------------------------------------------

    ax = axes[1]

    if val_acc:
        ax.plot(
            range(1, len(val_acc) + 1),
            val_acc,
            label="val_acc",
            linewidth=2,
        )

    if val_macro_f1:
        ax.plot(
            range(1, len(val_macro_f1) + 1),
            val_macro_f1,
            label="val_macro_f1",
            linewidth=2,
        )

    ax.set_title("Validation metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)

    if val_acc or val_macro_f1:
        ax.legend()

    _mark_best_epoch(ax, result)

    # ------------------------------------------------------------
    # Panel 3: Gradient norm
    # ------------------------------------------------------------

    ax = axes[2]

    if grad_norm:
        ax.plot(
            range(1, len(grad_norm) + 1),
            grad_norm,
            label="grad_norm",
            linewidth=2,
        )

    ax.set_title("Gradient norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.25)

    if grad_norm:
        ax.legend()

    _mark_best_epoch(ax, result)

    # ------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------

    fig.suptitle(
        _get_config_text(result),
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError(
            "results must contain at least one experiment."
        )

    fig, ax = plt.subplots(
        figsize=(9, 5),
    )

    plotted = False

    for result in results:
        history = _get_history(result, metric)

        if history is None or len(history) == 0:
            continue

        exp_id = result.get(
            "exp_id",
            f"experiment_{len(ax.lines) + 1}",
        )

        epochs = range(1, len(history) + 1)

        ax.plot(
            epochs,
            history,
            linewidth=2,
            label=str(exp_id),
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        raise ValueError(
            f"No experiment contains metric '{metric}'."
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title if title else f"{metric} comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)
