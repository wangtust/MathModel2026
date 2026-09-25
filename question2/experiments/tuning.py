"""
MRRF hyper-parameter tuning with Optuna.

Run:

python question2/run.py experiment --name tuning

Outputs:

outputs/q2/tuning/

├── study.db
├── trials.json
├── best_config.json
└── optuna.log
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import optuna


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "outputs" / "q2" / "tuning"

RUN_DIR = ROOT / "outputs" / "q2" / "runs" / "mrrf"


# 最大化 validation selection_score
DIRECTION = "maximize"


def run_trial(config, trial_id):

    output = (
        RUN_DIR /
        f"optuna_trial_{trial_id}"
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )


    cmd = [
        "python",
        "question2/run.py",
        "train",
        "--model",
        "mrrf",
        "--output",
        str(output),
        "--skip-predict",
    ]


    for key, value in config.items():

        cmd.extend(
            [
                f"--{key.replace('_','-')}",
                str(value)
            ]
        )


    print(
        "\nRunning trial:",
        " ".join(cmd),
        flush=True
    )


    subprocess.run(
        cmd,
        check=True
    )


    metrics_file = output / "metrics.json"


    with open(
        metrics_file,
        "r",
        encoding="utf-8"
    ) as f:

        metrics=json.load(f)


    score = metrics["selection_score"]


    return score



def objective(trial):


    config = {

        # 学习率
        "lr":
            trial.suggest_float(
                "lr",
                1e-5,
                5e-4,
                log=True
            ),


        # 隐藏维度
        "hidden_dim":
            trial.suggest_categorical(
                "hidden_dim",
                [
                    24,
                    48,
                    96,
                    128
                ]
            ),


        # batch
        "batch_size":
            trial.suggest_categorical(
                "batch_size",
                [
                    16,
                    32,
                    64
                ]
            ),


        # 重建损失权重
        "reconstruction_weight":
            trial.suggest_float(
                "reconstruction_weight",
                0.05,
                0.5
            ),


        # 一致性损失
        "consistency_weight":
            trial.suggest_float(
                "consistency_weight",
                0.0,
                0.5
            )
    }


    score = run_trial(
        config,
        trial.number
    )


    return score



def save_results(study):


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    trials=[]


    for t in study.trials:

        trials.append(
            {
                "number": t.number,
                "value": t.value,
                "params": t.params
            }
        )


    with open(
        OUTPUT_DIR/"trials.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            trials,
            f,
            indent=2,
            ensure_ascii=False
        )


    best={

        **study.best_params,

        "selection_score":
            study.best_value
    }


    with open(
        OUTPUT_DIR/"best_config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best,
            f,
            indent=2,
            ensure_ascii=False
        )


    print(
        "\nBest configuration:"
    )

    print(
        json.dumps(
            best,
            indent=2,
            ensure_ascii=False
        )
    )



def main():


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    study = optuna.create_study(

        study_name="mrrf_tuning",

        storage=
        f"sqlite:///{OUTPUT_DIR/'study.db'}",

        load_if_exists=True,

        direction=DIRECTION,


        sampler=
        optuna.samplers.TPESampler(
            seed=2026
        ),


        pruner=
        optuna.pruners.MedianPruner()
    )


    study.optimize(

        objective,

        n_trials=30

    )


    save_results(
        study
    )



if __name__=="__main__":

    main()