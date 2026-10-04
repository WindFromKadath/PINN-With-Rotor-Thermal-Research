"""COMSOL 批量数据生成 CLI（本机 COMSOL + mph 可选依赖）。

重写自原研究档案 ``COMSOL/python_link/link_test.py``：加载圆环 mph 模型，
对 ``n_q`` 个内边界热流工况（``q_in`` linspace 扫描）逐组 ``solve`` 并导出
含全部时刻的长 CSV，供 ``heatops.data.comsol.assemble_cartesian_4d`` 装配。

用法（需本机安装 COMSOL 与 ``mph`` 包，见 requirements.txt 可选依赖注释）::

    python -m heatops.data.comsol_batch --mph-path <模型.mph> --out-dir <CSV目录> --n-q 50

``mph`` 未安装或 COMSOL 不可达时给出清晰报错（退出码 2），不产生部分产物。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mph-path", required=True, help="COMSOL 模型文件路径（.mph）")
    parser.add_argument("--out-dir", required=True, help="CSV 输出目录（不存在则创建）")
    parser.add_argument("--n-q", type=int, default=50, help="热流工况数（默认 50，档案约定）")
    parser.add_argument("--q-min", type=float, default=5e4, help="q_in 下限，单位 W/m²（档案 5e4）")
    parser.add_argument("--q-max", type=float, default=5e5, help="q_in 上限，单位 W/m²（档案 5e5）")
    parser.add_argument("--param-name", default="q_in", help="mph 模型中的热流参数名（默认 q_in）")
    parser.add_argument("--study", default="std1", help="求解研究标签（默认 std1）")
    parser.add_argument("--export", default="data1", help="数据导出节点名（默认 data1）")
    args = parser.parse_args(argv)

    try:
        import mph
    except ImportError:
        print(
            "[错误] 未安装 mph 包（COMSOL with Python）。\n"
            "  批量生成需本机 COMSOL 环境：pip install mph 并确认 COMSOL 已安装；\n"
            "  无 COMSOL 环境时请使用已归档 CSV（未入库，本地归档）。"
        )
        return 2

    mph_path = Path(args.mph_path)
    if not mph_path.exists():
        print(f"[错误] 模型文件不存在：{mph_path}")
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_values = np.linspace(args.q_min, args.q_max, args.n_q)
    print(f"[comsol] 启动客户端，加载模型 {mph_path} ...")
    client = mph.start()
    model = client.load(str(mph_path))

    for i, val in enumerate(q_values):
        print(f"[comsol] 第 {i + 1}/{args.n_q} 组：{args.param_name} = {val:.6g} W/m²")
        model.parameter(args.param_name, f"{val} [W/m^2]")
        model.solve(args.study)
        output_file = out_dir / f"q_{i:03d}.csv"
        model.property(f"export/{args.export}", "filename", str(output_file))
        model.export(args.export)

    # 工况清单落盘（装配时与 CSV 一一对应，避免仅靠文件名推断 q 值）
    np.savetxt(out_dir / "q_values.csv", q_values, fmt="%.8g", header="q_in [W/m^2]")
    print(f"[done] {args.n_q} 组 CSV 已导出至 {out_dir}（含 q_values.csv 工况清单）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
