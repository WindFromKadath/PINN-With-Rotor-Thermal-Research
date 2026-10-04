"""1D 三模型横向对比入口（档案 R005 ``final_compare.py`` 重写）。

读取若干 run_id 的产物（``metrics.json`` 与 ``predictions_eval.npz``），
在**同一 eval 划分**上重算 RelL2（校验各 run 的数据缓存键一致），
画四联热图与末时刻剖面线对比::

    python -m heatops.cli.compare_pino_1d --runs E102 E103 E104 --out-dir artifacts/E104/compare

``--out-dir`` 缺省时取 ``artifacts/<最后一个 run_id>/compare``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from heatops.evaluation import relative_l2
from heatops.evaluation.plots import plot_multi_field_comparison, plot_profile_final_time


def _auto_label(run_id: str, config: dict) -> str:
    """按损失权重推断变体标签（档案 final_compare 命名风格），无法推断时用 run_id。"""
    weights = ((config or {}).get("extra") or {}).get("weights") or {}
    w_data = float(weights.get("data", 0.0))
    w_physics = sum(float(weights.get(k, 0.0)) for k in ("pde", "ic", "bc"))
    if w_data > 0 and w_physics == 0:
        return "Data-Driven"
    if w_data == 0 and w_physics > 0:
        return "Physics-Only"
    if w_data > 0 and w_physics > 0:
        return "Hybrid (PINO)"
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="参与对比的 run_id（如 E102 E103 E104）")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认 artifacts/<末位 run_id>/compare）")
    parser.add_argument("--sample-index", type=int, default=0, help="作图用的 eval 样本序号")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts") / args.runs[-1] / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    preds: dict[str, np.ndarray] = {}
    rel_l2: dict[str, float] = {}
    metrics_table: dict[str, dict] = {}
    data_keys: set[str] = set()
    x_np = t_np = u_true = None

    for run_id in args.runs:
        run_dir = Path("artifacts") / run_id
        npz_path = run_dir / "predictions_eval.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} 不存在：请先运行 run_pino_1d（run_id={run_id}）")
        payload = np.load(npz_path)
        config = {}
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        key = ((config.get("extra") or {}).get("data_cache_key")) or "?"
        data_keys.add(key)

        label = _auto_label(run_id, config)
        if label in preds:
            label = f"{label} ({run_id})"
        labels.append(label)
        preds[label] = payload["u_pred"][args.sample_index]

        # 同一 eval 划分上重算 RelL2（不直接转抄 metrics.json）
        rel_l2[label] = relative_l2(payload["u_true"], payload["u_pred"])
        metrics_table[label] = {
            "run_id": run_id,
            "rel_l2_eval_recomputed": rel_l2[label],
            "data_cache_key": key,
        }
        print(f"[eval] {label:<16s} run={run_id} RelL2(eval)={rel_l2[label]:.4%}")

        if x_np is None:
            x_np, t_np = payload["x"], payload["t"]
            u_true = payload["u_true"][args.sample_index]
        elif not (np.array_equal(x_np, payload["x"]) and np.array_equal(t_np, payload["t"])):
            raise ValueError(f"run {run_id} 的网格与首个 run 不一致，无法同图对比")

    if len(data_keys) > 1:
        raise ValueError(f"各 run 数据缓存键不一致 {sorted(data_keys)}：eval 划分不同，拒绝横向对比")

    plot_multi_field_comparison(
        x_np, t_np, u_true, preds, rel_l2,
        out_dir / "triple_model_comparison.png",
    )
    plot_profile_final_time(
        x_np, t_np, u_true, preds,
        out_dir / "profile_triple_comparison.png",
    )
    with open(out_dir / "compare_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_table, f, ensure_ascii=False, indent=2)
    print(f"[done] 对比产物已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
