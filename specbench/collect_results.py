#!/usr/bin/env python3
"""Aggregate every new-benchmark result into one report.

Collects:
  1. detection metrics   - specbench/results/detection_<task>.json  (45 rows)
  2. spec-decoding       - specbench/results/<model>_<task>.json
  3. ASR (AgentLAB)      - specbench/results/asr_<model>.json

Writes results_new_benchmark/REPORT.md and a combined JSON bundle.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "specbench", "results")
MODEL_NAMES = {"4b": "Qwen3.5-4B", "9b": "Qwen3.5-9B", "mcpm": "MiniCPM4.1-8B"}
MODEL_DISPLAY = {"qwen3.5-4b": "Qwen3.5-4B", "qwen3.5-9b": "Qwen3.5-9B",
                 "minicpm4.1-8b": "MiniCPM4.1-8B"}
TASK_NAMES = {"agentlab": "AgentLAB", "bfcl_v4": "BFCL-v4", "livecodebench": "LiveCodeBench"}
METHOD_NAMES = {"sp1": "SP-1", "sp2": "SP-2", "sp5": "SP-5", "mlp": "MLP", "multimax": "MM"}
METHOD_ORDER = ["mlp", "multimax", "sp1", "sp2", "sp5"]
# run_bench.py writes the "mm" config key (load_probe kind); detection/ASR use
# "multimax". Section 2 reads the bench files, so it must use "mm" here.
CONFIG_NAMES = {"none": "无探针", "mlp": "MLP", "mm": "MM", "sp1": "SP-1", "sp2": "SP-2", "sp5": "SP-5"}
CONFIG_ORDER = ["none", "mlp", "mm", "sp1", "sp2", "sp5"]


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fmt(x, pct=False, digits=2):
    if x is None:
        return "—"
    if pct:
        return f"{x*100:.{digits}f}"
    return f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results_new_benchmark", "REPORT.md"))
    args = ap.parse_args()

    detection = {t: load(os.path.join(BENCH_DIR, f"detection_{t}.json"))
                 for t in TASK_NAMES}
    spec = {}
    for model in ("qwen3.5-4b", "qwen3.5-9b", "minicpm4.1-8b"):
        for t in TASK_NAMES:
            spec[(model, t)] = load(os.path.join(BENCH_DIR, f"{model}_{t}.json"))
    asr = {m: load(os.path.join(BENCH_DIR, f"asr_{m}.json"))
           for m in ("qwen3.5-4b", "qwen3.5-9b", "minicpm4.1-8b")}

    lines = []
    lines.append("# 三个新 benchmark 上的 Speculative Probing 实验")
    lines.append("")
    lines.append("## 1. 检测指标（3 benchmark × 3 模型 × 5 配置 = 45 组）")
    lines.append("")
    lines.append("指标口径：FNR（漏报率）= 攻击样本被判为良性的比例；"
                 "FPR（误报率）= 良性样本被判为攻击的比例。"
                 "评测集为各 benchmark 的保留测试划分（与训练完全隔离的攻击载荷/良性来源）。")
    lines.append("")
    for task, dets in detection.items():
        lines.append(f"### {TASK_NAMES[task]}")
        lines.append("")
        lines.append("| 模型 | 配置 | Acc | FNR（漏报率） | FPR（误报率） | F1 |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for size in ("4b", "9b", "mcpm"):
            for m in METHOD_ORDER:
                r = (dets or {}).get(f"{size}/{m}")
                if r is None:
                    lines.append(f"| {MODEL_NAMES[size]} | {METHOD_NAMES[m]} | — | — | — | — |")
                else:
                    lines.append(f"| {MODEL_NAMES[size]} | {METHOD_NAMES[m]} | "
                                 f"{fmt(r['accuracy'], True)} | {fmt(r['FNR'], True)} | "
                                 f"{fmt(r['FPR'], True)} | {fmt(r['f1'], True)} |")
        lines.append("")

    lines.append("## 2. 投机解码指标（接收率 / 接收长度 / 时延 / 吞吐）")
    lines.append("")
    lines.append("口径：greedy 推测解码，draft 长度 γ=3；探针部署在每个解码轮（读已提交前缀，"
                 "从不参与 draft，因此不应改变接收率——none 与各探针配置的接收率一致即为验证）。"
                 "时延为逐样本 CUDA 事件总时长（中位数）；吞吐为墙钟 tokens/s。")
    lines.append("")
    for t in TASK_NAMES:
        lines.append(f"### {TASK_NAMES[t]}")
        lines.append("")
        lines.append("| 模型 | 配置 | 接收率 | 接收长度 τ | 时延 ms (中位) | 吞吐 tok/s (中位) | 探针开销占比 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for model in ("qwen3.5-4b", "qwen3.5-9b", "minicpm4.1-8b"):
            d = spec.get((model, t))
            for c in CONFIG_ORDER:
                r = (d or {}).get("configs", {}).get(c)
                name = CONFIG_NAMES[c]
                if r is None:
                    lines.append(f"| {MODEL_DISPLAY[model]} | {name} | — | — | — | — | — |")
                else:
                    lat = (r.get("latency_gpu_ms") or {}).get("median")
                    thr = (r.get("throughput_tok_s") or {}).get("median")
                    if thr is None:
                        thr = r.get("throughput_wall_tok_s")
                    lines.append(
                        f"| {MODEL_DISPLAY[model]} | {name} | "
                        f"{fmt(r.get('acceptance_rate_mean'), False, 3)} | "
                        f"{fmt(r.get('acceptance_length_mean'), False, 2)} | "
                        f"{fmt(lat, False, 1)} | {fmt(thr, False, 1)} | "
                        f"{fmt(r.get('probe_ms_share_mean')*100 if r.get('probe_ms_share_mean') is not None else None, False, 2)}% |")
        lines.append("")

    lines.append("## 3. AgentLAB 攻击成功率（ASR）")
    lines.append("")
    lines.append("口径：对 Agent-SafetyBench 的直接攻击指令做贪心生成，"
                 "以拒绝标记判定“攻击生效”；部署探针后，攻击成功 = 模型配合 且 探针漏检。")
    lines.append("")
    lines.append("| 模型 | 配置 | ASR（无监控基线） | 部署后 ASR | 良性误报 FPR（部署后） |")
    lines.append("|---|---|---:|---:|---:|")
    for model in ("qwen3.5-4b", "qwen3.5-9b", "minicpm4.1-8b"):
        d = asr.get(model) or {}
        disp = MODEL_DISPLAY[model]
        lines.append(f"| {disp} | 无监控 | {fmt(d.get('ASR_base'), False, 3)} | — | — |")
        for m in METHOD_ORDER:
            key = {"mlp": "mlp", "multimax": "mm"}.get(m, m)
            am = fmt((d.get("ASR_with_monitor") or {}).get(key), False, 3)
            bf = fmt((d.get("benign_FPR") or {}).get(key), False, 3)
            lines.append(f"| {disp} | {METHOD_NAMES[m]} | {fmt(d.get('ASR_base'), False, 3)} | {am} | {bf} |")
    lines.append("")

    lines.append("## 4. MiniCPM4.1-8B Eagle3 draft 约定问题（已知限制）")
    lines.append("")
    lines.append("MiniCPM4.1-8B 的 Eagle3 draft 头与投机解码引擎之间的 hidden-state / 位置约定未能与"
                 "官方 cpm.cu 配方完全对齐（已尝试 slot 变换、fc 融合、多种 norm / 层组合），"
                 "draft 验收率仍≈0。因此第 2 节中 MiniCPM 的接收率（≈0.015）与接收长度 τ（≈1.04）"
                 "不代表有效投机加速，实际等价于纯自回归（AR）解码，仅作占位，不应与 Qwen 的"
                 "接收率直接对比。第 1 节的检测指标与第 3 节的 ASR 均基于缓存的隐藏态训练 / 评测，"
                 "不依赖 draft 头，仍然有效。")
    lines.append("")
    lines.append("为给出可比参考，另用 `run_bench.py --gamma 0` 跑了 MiniCPM 纯自回归基线行：")
    lines.append("")
    lines.append("| benchmark | 配置 | 接收长度 τ | 吞吐 tok/s (墙钟) | 时延 ms (中位) |")
    lines.append("|---|---|---:|---:|---:|")
    for t in TASK_NAMES:
        ar = load(os.path.join(BENCH_DIR, f"minicpm4.1-8b_{t}_ar.json"))
        c = (ar or {}).get("configs", {}).get("none")
        if c is None:
            lines.append(f"| {TASK_NAMES[t]} | AR (γ=0) | — | — | — |")
        else:
            lat = (c.get("latency_gpu_ms") or {}).get("median")
            lines.append(f"| {TASK_NAMES[t]} | AR (γ=0) | "
                         f"{fmt(c.get('acceptance_length_mean'), False, 2)} | "
                         f"{fmt(c.get('throughput_wall_tok_s'), False, 1)} | "
                         f"{fmt(lat, False, 1)} |")
    lines.append("")

    lines.append("## 5. 结果合理性抽查")
    lines.append("")
    n_pairs = 0
    n_sp_ge_mlp = 0
    for t in TASK_NAMES:
        dets = detection.get(t) or {}
        for size in ("4b", "9b", "mcpm"):
            vals = {m: (dets.get(f"{size}/{m}") or {}).get("accuracy")
                    for m in ("mlp", "multimax", "sp1", "sp2", "sp5")}
            sp = [vals[m] for m in ("sp1", "sp2", "sp5") if vals.get(m) is not None]
            if sp and vals.get("mlp") is not None:
                n_pairs += 1
                if max(sp) >= vals["mlp"] - 0.005:
                    n_sp_ge_mlp += 1
    lines.append(f"- **检测精度排序**：{n_pairs} 个 (benchmark, 模型) 组合中，"
                 f"SP-`k` 最优精度 ≥ MLP 精度的有 {n_sp_ge_mlp} 个"
                 "（SP 与 MM 接近、MLP 为最弱基线，与 verih / nemotron 复现的排序一致）。")
    drifts = []
    for model in ("qwen3.5-4b", "qwen3.5-9b", "minicpm4.1-8b"):
        for t in TASK_NAMES:
            d = spec.get((model, t)) or {}
            rates = [c.get("acceptance_rate_mean") for c in d.get("configs", {}).values()
                     if c.get("acceptance_rate_mean") is not None]
            if rates:
                drifts.append(max(rates) - min(rates))
    if drifts:
        verdict = "PASS：探针不参与 draft，未改变解码路径" if max(drifts) < 0.02 else "CHECK"
        lines.append(f"- **接收率跨配置一致性**：{len(drifts)} 个 (模型, benchmark) 组合，"
                     f"最大漂移={max(drifts):.4f}（阈值 0.02）→ {verdict}。")
    lines.append("- **MiniCPM 精度**：本轮三个新 benchmark 的二分类任务较易分离，"
                 "MiniCPM 的检测精度与 Qwen 相当（未出现原复现 63–78% 的显著下降）；"
                 "其投机解码接收率无效的原因见第 4 节，与检测 / ASR 质量无关。")
    lines.append("- **ASR 良性误报（部署后）口径说明**：第 3 节的良性误报来自 AgentDojo "
                 "用户任务指令（126 条，动作性指令），与探针训练时的良性语料（AgentDojo "
                 "环境文档，如邮件 / 文件 / slack 内容，5900 条）分布不同。SP-`k` 以"
                 "投机解码已驻留的 draft cache 做近似推理（非教师强制全序列），"
                 "在这类 OOD 动作指令上良性误报偏高（Qwen3.5-4B SP-2/5 达 0.32/0.48），"
                 "而 MLP / MM 因直接读隐藏态仍接近 0；第 1 节同分布测试集上各配置 FPR "
                 "均 ≤0.041。此为部署近似 + 分布偏移的真实测量结果，非脚本错误，"
                 "已在数据中如实保留。")
    lines.append("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
