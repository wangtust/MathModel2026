"""Rate-by-position validation grid for the selected MRRF model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from common.plotting import DOUBLE_COLUMN, INK, MUTED, apply_paper_style, panel_title, save_figure
from question2.data.dataset import FeatureStore
from question2.training.trainer import load_trained, make_loader
from question2.config import DATA
from question2.evaluation.evaluator import evaluate, public_metrics


ROOT=Path(__file__).resolve().parent
from common.paths import Q2_OUTPUT_DIR
OUT=Q2_OUTPUT_DIR / "experiments" / "robustness"
SCENARIOS=("text","audio","vision","text+audio","text+vision","audio+vision")
NAMES=("T","A","V","T+A","T+V","A+V")
POSITIONS=("beginning","middle","end")
RATES=(.1,.2,.3,.4,.5)


def _cell(value: float) -> str:
    return "0.00" if abs(value)<.005 else f"{value:+.2f}"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir",type=Path,default=Q2_OUTPUT_DIR / "runs" / "mrrf" / "full" / "seed_2026")
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    apply_paper_style(chinese=True)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,checkpoint=load_trained(args.model_dir,device)
    loader=make_loader(FeatureStore(DATA),"valid",64,device)
    clean=evaluate(model,loader,device,"clean")
    rows=[]
    for rate in RATES:
        for scenario in SCENARIOS:
            for position in POSITIONS:
                result=evaluate(model,loader,device,scenario,rate=rate,position=position)
                rows.append({"checkpoint_variant":checkpoint["variant"],
                             "checkpoint_seed":checkpoint["seed"],
                             **public_metrics(result),
                             "delta_macro_f1":result["macro_f1"]-clean["macro_f1"],
                             "delta_mae":result["mae"]-clean["mae"]})
        print(f"Position grid complete: {rate:.0%}",flush=True)
    with (out / "position_rate_grid.csv").open(
            "w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

    # Attachment 3 may contain A/V gaps with different starts and lengths.
    # Audit this shift separately from the fixed-rate position grid above.
    layout_rows=[]
    for layout, rate_mode, rate in (("shared", "fixed_30_percent", .3),
                                    ("independent", "fixed_30_percent", .3),
                                    ("independent", "per_modality_10_to_50_percent", None)):
        result=evaluate(model,loader,device,"audio+vision",rate=rate,
                        gap_layout=layout)
        layout_rows.append({"gap_layout":layout,"rate_mode":rate_mode,
                            "checkpoint_seed":checkpoint["seed"],
                            **public_metrics(result),
                            "delta_macro_f1":result["macro_f1"]-clean["macro_f1"],
                            "delta_mae":result["mae"]-clean["mae"]})
    with (out / "gap_layout_validation.csv").open(
            "w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(layout_rows[0]))
        writer.writeheader();writer.writerows(layout_rows)

    panels=[]
    for rate in RATES:
        lookup={(r["scenario"],r["position"]):r["delta_mae"]
                for r in rows if np.isclose(r["missing_rate"],rate)}
        panels.append(np.array([[lookup[(scenario,position)] for position in POSITIONS]
                                for scenario in SCENARIOS]))
    bound=max(.03,max(float(np.max(np.abs(x))) for x in panels))

    # Show all five measured missing-rate levels. The previous implementation
    # accidentally created three axes and silently truncated the 40%/50% panels.
    fig,axes=plt.subplots(1,len(RATES),figsize=(DOUBLE_COLUMN,3.55),sharey=True,
                          gridspec_kw={"wspace":.08},constrained_layout=True)
    image=None
    for k,(ax,rate,data) in enumerate(zip(axes,RATES,panels)):
        image=ax.imshow(data,cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),
                        aspect="auto")
        panel_title(ax,chr(ord('a')+k),f"{rate:.0%}",pad=4)
        ax.set_xticks(range(3),["Beg.","Mid.","End"])
        ax.set_yticks(range(6),NAMES)
        ax.set_xlabel("缺失位置")
        if k==0: ax.set_ylabel("缺失模态")
        ax.tick_params(length=0,labelsize=6.2)
        for i in range(6):
            for j in range(3):
                ax.text(j,i,_cell(data[i,j]),ha="center",va="center",fontsize=5.2,
                        color="white" if abs(data[i,j])>.65*bound else INK)
        for side in ax.spines.values(): side.set_visible(False)
    fig.colorbar(image,ax=axes,shrink=.72,pad=.025,label="Δ MAE = MAE(masked) − MAE(clean)")
    fig.text(.5,-.025,
             "完整展示 10%–50% 五档缺失比例，共 5×6×3=90 个验证扰动条件；原生全零与人工遮挡独立记录。",
             ha="center",fontsize=6.1,color=MUTED)
    save_figure(fig,out,"fig19_position_rate_interaction")

    main_rows=[r for r in rows if r["scenario"]=="text"]
    worst=max(main_rows,key=lambda r:r["delta_mae"])
    rate_text="、".join(f"{r:.0%}" for r in RATES)
    doc=["# 缺失比例×位置补充实验", "",
         f"对 MRRF {checkpoint['variant']}（种子 {checkpoint['seed']}）使用附件二固定验证集，在 {rate_text} 五档连续缺失比例下，测试六类单/双模态缺失在开头、中间、结尾的位置。",
         f"共 {len(rows)} 条条件（5×6×3=90），clean Macro-F1={clean['macro_f1']:.3f}、MAE={clean['mae']:.3f}。",
         f"文本缺失中 MAE 增幅最大的一格为 {worst['missing_rate']:.0%}、{worst['position']}，ΔMAE={worst['delta_mae']:+.3f}；不同模态组合应分别读取 `position_rate_grid.csv`。",
         "图 `fig19_position_rate_interaction.pdf` 完整展示 10%–50% 五档 ΔMAE；同一表还含 Macro-F1、Accuracy、Pearson 及其 clean 差值。",
         "本图为一个固定检查点上的验证扰动分析，不是三个训练种子的误差区间；不据此更换模型。"]
    (out / "position_rate_results.md").write_text("\n".join(doc)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
