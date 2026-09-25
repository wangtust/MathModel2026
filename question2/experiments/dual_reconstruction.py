"""Matched-seed, fixed-mask audit of the exploratory dual-reconstruction variant."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from question2.data.dataset import FeatureStore
from question2.training.trainer import load_trained, make_loader
from question2.config import DATA
from question2.evaluation.evaluator import evaluate, public_metrics


ROOT = Path(__file__).resolve().parents[2]
from question2.config import run_dir
from common.paths import Q2_OUTPUT_DIR
OUT = Q2_OUTPUT_DIR / "experiments" / "ablation"
SCENARIOS = ("text", "audio", "vision", "text+audio", "text+vision", "audio+vision")
RATES = (.1, .2, .3, .4, .5, .6)
LABELS = ("T", "A", "V", "T+A", "T+V", "A+V")


def _cell(value: float) -> str:
    return "0.00" if abs(value) < .005 else f"{value:+.2f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],
                         "font.size":8,"pdf.fonttype":42,"ps.fonttype":42})
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = FeatureStore(DATA)
    loader = make_loader(store, "valid", 64, device)
    rows = []
    reports = {}
    for variant, directory in (("full", run_dir("mrrf", 2026, "full")),
                               ("dual_recon", run_dir("mrrf", 2026, "dual_recon"))):
        model, checkpoint = load_trained(directory, device)
        model.eval()
        reports[variant] = json.loads((directory / "metrics.json").read_text())
        cases = [("clean", 0., "random")]
        cases += [(scenario, rate, "random") for scenario in SCENARIOS for rate in RATES]
        cases += [(scenario, .3, position) for scenario in SCENARIOS
                  for position in ("beginning", "middle", "end")]
        for scenario, rate, position in cases:
            result = evaluate(model, loader, device, scenario,
                              rate=rate, position=position)
            rows.append({"variant": variant, "checkpoint_seed": checkpoint["seed"],
                         **public_metrics(result)})
        print(f"Evaluated {variant}: {len(cases)} conditions", flush=True)
        del model

    path = OUT / "dual_reconstruction_grid.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    seed_rows=[]
    for variant in ("full", "dual_recon"):
        for seed in (2026,2027,2028):
            directory=(run_dir("mrrf", seed, variant))
            data=json.loads((directory / "metrics.json").read_text())
            seed_rows.append({"variant":variant,"seed":seed,
                              "selection_score":float(data["selection_score"]),
                              "valid_clean_macro_f1":float(data["validation"]["clean"]["macro_f1"]),
                              "valid_clean_mae":float(data["validation"]["clean"]["mae"]),
                              "test_clean_macro_f1":float(data["heldout_test"]["clean"]["macro_f1"]),
                              "test_clean_mae":float(data["heldout_test"]["clean"]["mae"])})
    with (OUT / "dual_reconstruction_three_seed.csv").open(
            "w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(seed_rows[0]))
        writer.writeheader();writer.writerows(seed_rows)

    lookup = {(r["variant"], r["scenario"], float(r["missing_rate"]), r["position"]): r
              for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.85))
    fig.subplots_adjust(left=.12, right=.90, bottom=.19, top=.80, wspace=.27)
    fig.suptitle("双路重建相对原 MRRF 的验证集变化", x=.12, y=.98,
                 ha="left", fontsize=10.5, fontweight="semibold")
    fig.text(.12, .89, "同一训练种子 2026、同一验证集、同一固定连续遮挡；正值表示双路重建较优。",
             fontsize=7.2, color="#4e5c63")
    matrices = []
    for metric in ("macro_f1", "mae"):
        values = np.empty((len(SCENARIOS), len(RATES)))
        for i, scenario in enumerate(SCENARIOS):
            for j, rate in enumerate(RATES):
                full = lookup[("full", scenario, rate, "random")][metric]
                dual = lookup[("dual_recon", scenario, rate, "random")][metric]
                values[i, j] = (dual - full) if metric == "macro_f1" else (full - dual)
        matrices.append(values)
    limit = max(.04, max(float(np.max(np.abs(m))) for m in matrices))
    for ax, data, title in zip(axes, matrices,
                               ("(a) Macro-F1 收益", "(b) MAE 降低量")):
        image = ax.imshow(data, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),
                          aspect="auto")
        ax.set_title(title, loc="left", fontsize=8.4, pad=6)
        ax.set_xticks(range(len(RATES)), [f"{int(r*100)}" for r in RATES])
        ax.set_yticks(range(len(SCENARIOS)), LABELS)
        ax.set_xlabel("连续缺失比例 / %", fontsize=7.4)
        ax.tick_params(length=0, labelsize=7)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, _cell(data[i,j]), ha="center", va="center",
                        fontsize=5.6, color="white" if abs(data[i,j])>.65*limit else "#273940")
        for side in ax.spines.values():side.set_visible(False)
    cax=fig.add_axes((.92,.19,.016,.61))
    fig.colorbar(image,cax=cax,label="双路相对原模型的收益")
    fig.text(.12,.055,"仅为新增结构的单种子探索实验；选模规则与问题二主实验一致，留出测试未参与选择。",
             fontsize=6.6,color="#59666c")
    fig.savefig(OUT / "fig18_dual_reconstruction.pdf",bbox_inches="tight")
    fig.savefig(OUT / "fig18_dual_reconstruction.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

    def val(variant,scenario,rate,metric):
        return float(lookup[(variant,scenario,rate,"random")][metric])
    def summary(variant,metric):
        numbers=np.array([r[metric] for r in seed_rows if r["variant"]==variant])
        return f"{numbers.mean():.3f} ± {numbers.std(ddof=1):.3f}"
    doc = ["# 双路重建探索实验", "",
           "新增变体在不覆盖主模型检查点的条件下，把每个模态的掩码自注意力编码与 MulT 跨模态序列重建融合。",
           "仅在人工遮挡且原生可用的位置计算重建损失；输入、划分、选模规则与原 MRRF 完全相同。", "",
           "| 指标 | 原 MRRF | 双路重建 |", "|---|---:|---:|",
           f"| 三种子验证选模分数 | {summary('full','selection_score')} | {summary('dual_recon','selection_score')} |",
           f"| 三种子验证 clean Macro-F1 | {summary('full','valid_clean_macro_f1')} | {summary('dual_recon','valid_clean_macro_f1')} |",
           f"| 三种子留出测试 clean Macro-F1 | {summary('full','test_clean_macro_f1')} | {summary('dual_recon','test_clean_macro_f1')} |",
           f"| 种子 2026 验证选模分数 | {reports['full']['selection_score']:.3f} | {reports['dual_recon']['selection_score']:.3f} |",
           f"| 种子 2026 验证 clean Macro-F1 | {val('full','clean',0,'macro_f1'):.3f} | {val('dual_recon','clean',0,'macro_f1'):.3f} |",
           f"| 种子 2026 验证 clean MAE | {val('full','clean',0,'mae'):.3f} | {val('dual_recon','clean',0,'mae'):.3f} |",
           f"| 种子 2026 验证文本缺失 60% Macro-F1 | {val('full','text',.6,'macro_f1'):.3f} | {val('dual_recon','text',.6,'macro_f1'):.3f} |",
           f"| 种子 2026 验证文本缺失 60% MAE | {val('full','text',.6,'mae'):.3f} | {val('dual_recon','text',.6,'mae'):.3f} |",
           f"| 种子 2026 留出测试 clean Macro-F1 | {reports['full']['heldout_test']['clean']['macro_f1']:.3f} | {reports['dual_recon']['heldout_test']['clean']['macro_f1']:.3f} |",
           f"| 种子 2026 留出测试 clean MAE | {reports['full']['heldout_test']['clean']['mae']:.3f} | {reports['dual_recon']['heldout_test']['clean']['mae']:.3f} |",
           "", "表中未注明‘三种子’的行均为种子 2026 的条件对照；三种子逐项数值见 `dual_reconstruction_three_seed.csv`。",
           "全 55 条固定验证条件见 `dual_reconstruction_grid.csv`，缺失比例对比见 `fig18_dual_reconstruction.pdf`。",
           "这是结构补全后的探索实验；图 18 的场景热图只有种子 2026，三种子均值也不能证明结构优越性。附件三仍采用既有预先确定规则选定的主模型。"]
    (OUT / "dual_reconstruction_results.md").write_text("\n".join(doc)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
