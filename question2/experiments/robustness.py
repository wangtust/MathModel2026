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
    plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],
                         "font.size":8,"pdf.fonttype":42,"ps.fonttype":42})
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
    fig,axes=plt.subplots(1,3,figsize=(7.4,3.7))
    fig.subplots_adjust(left=.10,right=.89,bottom=.19,top=.80,wspace=.13)
    fig.suptitle("缺失比例与缺失位置的交互：验证集 MAE 变化",x=.10,y=.98,
                 ha="left",fontsize=10.5,fontweight="semibold")
    fig.text(.10,.89,f"MRRF {checkpoint['variant']} · 种子 {checkpoint['seed']} · n=728；每格相对同一模型 clean MAE，正值表示误差增加。",
             fontsize=7.0,color="#4d5d64")
    for ax,rate,data in zip(axes,RATES,panels):
        image=ax.imshow(data,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),
                        aspect="auto")
        ax.set_title(f"缺失 {rate:.0%}",fontsize=8.4,pad=6)
        ax.set_xticks(range(3),["开头","中间","结尾"])
        ax.set_yticks(range(6),NAMES)
        ax.tick_params(length=0,labelsize=6.7)
        for i in range(6):
            for j in range(3):
                ax.text(j,i,_cell(data[i,j]),ha="center",va="center",fontsize=6,
                        color="white" if abs(data[i,j])>.65*bound else "#263940")
        for side in ax.spines.values():side.set_visible(False)
    cax=fig.add_axes((.91,.19,.017,.61))
    fig.colorbar(image,cax=cax,label="MAE − clean MAE")
    fig.text(.10,.055,"缺失区间按各样本有效正文长度计算；原生全零位置与注入遮挡独立记录。",
             fontsize=6.7,color="#52636a")
    fig.savefig(out / "fig19_position_rate_interaction.pdf",bbox_inches="tight")
    fig.savefig(out / "fig19_position_rate_interaction.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

    main_rows=[r for r in rows if r["scenario"]=="text"]
    worst=max(main_rows,key=lambda r:r["delta_mae"])
    doc=["# 缺失比例×位置补充实验", "",
         f"对 MRRF {checkpoint['variant']}（种子 {checkpoint['seed']}）使用附件二固定验证集，分别在 20%、30%、40% 缺失比例下，测试六类单/双模态缺失在开头、中间、结尾的位置。",
         f"共 {len(rows)} 条条件，clean Macro-F1={clean['macro_f1']:.3f}、MAE={clean['mae']:.3f}。",
         f"文本缺失中 MAE 增幅最大的一格为 {worst['missing_rate']:.0%}、{worst['position']}，ΔMAE={worst['delta_mae']:+.3f}；不同模态组合应分别读取 `position_rate_grid.csv`。",
         "图 `fig19_position_rate_interaction.pdf` 展示 ΔMAE；同一表还含 Macro-F1、Accuracy、Pearson 及其 clean 差值。",
         "本补充图为一个已选检查点上的验证扰动分析，不是三个训练种子的误差区间；不据此更换模型。"]
    (out / "position_rate_results.md").write_text("\n".join(doc)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
