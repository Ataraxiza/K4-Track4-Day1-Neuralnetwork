"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")

    cfg = result.get("cfg")
    if not isinstance(cfg, dict):
        raise ValueError("result must contain a dict under 'cfg'")

    exp_id = cfg.get("exp_id")
    if exp_id is None:
        # Some experiment runners may keep exp_id at the top level.
        exp_id = result.get("exp_id")

    if exp_id is None:
        raise ValueError("result['cfg'] must contain 'exp_id'")

    history = result.get("history", {})
    summary = result.get("summary", {})

    if not isinstance(history, dict):
        raise TypeError("result['history'] must be a dict")

    if not isinstance(summary, dict):
        raise TypeError("result['summary'] must be a dict")

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{exp_id}.json"

    # Explicitly construct the payload so best_state cannot accidentally
    # end up in the JSON file.
    payload = {
        "cfg": cfg,
        "history": history,
        "summary": summary,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    return str(output_path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    results_path = Path(results_dir)

    if not results_path.exists():
        return []

    if not results_path.is_dir():
        raise NotADirectoryError(
            f"Results path is not a directory: {results_path}"
        )

    results: list[dict] = []

    for path in results_path.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)

        if not isinstance(result, dict):
            raise ValueError(f"Invalid result file (expected dict): {path}")

        cfg = result.get("cfg")
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Invalid result file (missing dict 'cfg'): {path}"
            )

        if "exp_id" not in cfg:
            raise ValueError(
                f"Invalid result file (cfg.exp_id missing): {path}"
            )

        results.append(result)

    # Sort by exp_id. Using str() makes sorting robust if exp_id is
    # accidentally represented as an int in some result files.
    results.sort(
        key=lambda result: str(result["cfg"]["exp_id"])
    )

    return results


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")

    cfg = result.get("cfg", {})
    summary = result.get("summary", {})

    if not isinstance(cfg, dict):
        raise TypeError("result['cfg'] must be a dict")

    if not isinstance(summary, dict):
        raise TypeError("result['summary'] must be a dict")

    exp_id = cfg.get("exp_id")

    if exp_id is None:
        exp_id = result.get("exp_id")

    if exp_id is None:
        raise ValueError("Experiment result does not contain exp_id")

    if eval_scores is not None and not isinstance(eval_scores, dict):
        raise TypeError("eval_scores must be a dict or None")

    # Merge cfg first, then summary. This lets summary values take
    # precedence if both happen to contain the same key.
    merged: dict[str, Any] = {}
    merged.update(cfg)
    merged.update(summary)

    # Evaluation metrics are deliberately supplied separately because
    # only selected experiments are evaluated on the external/eval set.
    if eval_scores is not None:
        merged.update(eval_scores)

    merged["exp_id"] = exp_id
    merged["figure_file"] = f"figures/{exp_id}.png"
    merged["notes"] = notes

    # Return exactly the columns that are data fields in the requested
    # table. Formula columns are intentionally not included.
    row = {
        column: merged.get(column)
        for column in DATA_COLUMNS
    }

    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to write the Excel file. "
            "Install it with: pip install openpyxl"
        ) from exc

    if not isinstance(rows, list):
        raise TypeError("rows must be a list")

    template = Path(template_path)
    output = Path(out_path)

    if not template.exists():
        raise FileNotFoundError(
            f"Excel template not found: {template}"
        )

    # IMPORTANT:
    # Do not use data_only=True because we need to preserve formulas.
    wb = load_workbook(template)

    if "Experiments" not in wb.sheetnames:
        raise KeyError(
            'Template does not contain a sheet named "Experiments"'
        )

    ws = wb["Experiments"]

    # Read headers from row 1.
    headers: dict[str, int] = {}

    for column_index in range(1, ws.max_column + 1):
        value = ws.cell(row=1, column=column_index).value

        if value is None:
            continue

        header = str(value).strip()

        if header in headers:
            raise ValueError(
                f"Duplicate column name in template header: {header!r}"
            )

        headers[header] = column_index

    if not headers:
        raise ValueError(
            'Sheet "Experiments" does not contain a header row'
        )

    # Check that all required data columns exist.
    missing_columns = [
        column
        for column in DATA_COLUMNS
        if column not in headers
    ]

    if missing_columns:
        raise ValueError(
            "Template is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Make sure the formula columns exist as well. We do not write to them,
    # but their presence is useful for catching an incorrect template.
    missing_formula_columns = [
        column
        for column in FORMULA_COLUMNS
        if column not in headers
    ]

    if missing_formula_columns:
        raise ValueError(
            "Template is missing expected formula columns: "
            + ", ".join(missing_formula_columns)
        )

    # Write rows starting at Excel row 2.
    for row_index, row_data in enumerate(rows, start=2):
        if not isinstance(row_data, dict):
            raise TypeError(
                f"Row {row_index} must be a dict, got "
                f"{type(row_data).__name__}"
            )

        for key, value in row_data.items():
            # Never write formula columns.
            if key in FORMULA_COLUMNS:
                continue

            # Ignore unknown keys rather than accidentally adding columns
            # that aren't part of the template.
            if key not in headers:
                continue

            column_index = headers[key]
            ws.cell(
                row=row_index,
                column=column_index,
                value=value,
            )

    # Ensure the output directory exists when an output directory was given.
    output.parent.mkdir(parents=True, exist_ok=True)

    wb.save(output)
