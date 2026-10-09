"""
Detection trade-off figure for the revised manuscript
=====================================================
Reads the saved out-of-fold predictions of run_adl_error_detection.py (no
retraining) and writes:

  fig_tradeoff.pdf  (a) recall vs. false-positive rate: ROC curves of the
                    continuous-score methods, with the rules and the OR fusion
                    as single operating points (participant-bootstrap 95% CIs);
                    (b) per task, error-condition episodes caught by the rules,
                    added by the GRU in the fusion, or missed, and false alarms
                    by source
  tradeoff.json     every number drawn in the figure

ROC curves trace all thresholds on the pooled out-of-fold scores; a point on a
curve other than the method's default cutoff is therefore chosen post hoc.

Usage:
  python -m scripts.plot_tradeoff --results <outputs/adl_errors> \
      --derived <results_rev1> --dest <latex/generated>
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from adl_nesy.utils.logging import setup_logging

log = logging.getLogger(__name__)

SEED = 42
N_BOOT = 2000
TASKS = ["PhoneCall", "WashHands", "Cook", "Eat", "Clean"]

# Same palette slots as the v2 figure; markers and line styles give a second,
# colour-independent encoding.
CURVES = {
    "RF": ("Random forest", "#1baf7a", "s", "--"),
    "LogReg": ("Logistic regression", "#7d5bbe", "P", (0, (3, 1, 1, 1))),
    "GRU": ("GRU", "#eda100", "^", "-."),
    "IForest": ("Isolation forest", "#eb6834", "v", (0, (1, 1))),
}
POINTS = {
    "Rules": ("Procedural rules", "#2a78d6", "o"),
    "NeSy": (r"Rules $\vee$ GRU", "#e87ba4", "D"),
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
MISSED = "#d9d8d2"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot the detection trade-off figure.")
    ap.add_argument("--results", type=str, required=True, help="Original outputs/adl_errors (read-only)")
    ap.add_argument("--derived", type=str, required=True, help="Where tradeoff.json is written")
    ap.add_argument("--dest", type=str, required=True, help="Where fig_tradeoff.pdf is written")
    return ap.parse_args()


def _rates(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    return float(pred[y == 0].mean()), float(pred[y == 1].mean())


def _bootstrap_rates(df: pd.DataFrame, col: str) -> Dict[str, List[float]]:
    """Participant-cluster percentile CIs for (FPR, recall), as in the main runner."""
    rng = np.random.default_rng(SEED)
    groups = {p: g for p, g in df.groupby("participant")}
    keys = np.array(list(groups.keys()))
    fprs, recs = [], []
    for _ in range(N_BOOT):
        d = pd.concat([groups[k] for k in rng.choice(keys, size=len(keys), replace=True)], ignore_index=True)
        f, r = _rates(d["y"].to_numpy(), d[col].to_numpy())
        fprs.append(f)
        recs.append(r)
    return {"fpr_ci95": [float(v) for v in np.percentile(fprs, [2.5, 97.5])],
            "recall_ci95": [float(v) for v in np.percentile(recs, [2.5, 97.5])]}


def compute(df: pd.DataFrame) -> Dict[str, Any]:
    y = df["y"].to_numpy()
    out: Dict[str, Any] = {"n_error": int(y.sum()), "n_correct": int((y == 0).sum()), "points": {}, "curves": {}}
    for meth in POINTS:
        f, r = _rates(y, df[f"pred_{meth}"].to_numpy())
        out["points"][meth] = {"fpr": f, "recall": r, **_bootstrap_rates(df, f"pred_{meth}")}
    rules_fpr = out["points"]["Rules"]["fpr"]
    for meth in CURVES:
        fpr, tpr, _ = roc_curve(y, df[f"score_{meth}"])
        f, r = _rates(y, df[f"pred_{meth}"].to_numpy())
        out["curves"][meth] = {
            "auroc": float(roc_auc_score(y, df[f"score_{meth}"])),
            "default_fpr": f, "default_recall": r,
            # Highest recall on the curve without exceeding the rules' FPR (post hoc)
            "recall_at_rules_fpr": float(tpr[fpr <= rules_fpr + 1e-12].max()),
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        }
    per_task = {}
    for task in TASKS:
        t = df[df["task_name"] == task]
        err, ok = t[t["y"] == 1], t[t["y"] == 0]
        gru_only = lambda d: int(((d["pred_GRU"] == 1) & (d["pred_Rules"] == 0)).sum())
        per_task[task] = {
            "n_error": len(err), "n_correct": len(ok),
            "tp_rules": int(err["pred_Rules"].sum()), "tp_gru_added": gru_only(err),
            "fp_rules": int(ok["pred_Rules"].sum()), "fp_gru_added": gru_only(ok),
        }
        per_task[task]["missed_by_fusion"] = len(err) - per_task[task]["tp_rules"] - per_task[task]["tp_gru_added"]
    out["per_task"] = per_task
    return out


def plot(res: Dict[str, Any], dest: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.8, 2.45), gridspec_kw={"width_ratios": [1.0, 1.08]})

    # (a) operating points
    for meth, (label, color, marker, ls) in CURVES.items():
        c = res["curves"][meth]
        ax.plot(c["fpr"], c["tpr"], color=color, linestyle=ls, linewidth=1.1, drawstyle="steps-post",
                label=f"{label} ({c['auroc']:.2f})", zorder=2)
        ax.plot(c["default_fpr"], c["default_recall"], marker=marker, markersize=4, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.0, linestyle="none", zorder=3)
    for meth, (label, color, marker) in POINTS.items():
        p = res["points"][meth]
        ax.errorbar(p["fpr"], p["recall"],
                    xerr=[[p["fpr"] - p["fpr_ci95"][0]], [p["fpr_ci95"][1] - p["fpr"]]],
                    yerr=[[p["recall"] - p["recall_ci95"][0]], [p["recall_ci95"][1] - p["recall"]]],
                    fmt=marker, color=color, markersize=4.5, elinewidth=0.8, capsize=1.5, label=label, zorder=4)
    r, n = res["points"]["Rules"], res["points"]["NeSy"]
    ax.annotate("", xy=(n["fpr"], n["recall"]), xytext=(r["fpr"], r["recall"]),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8, "shrinkA": 4, "shrinkB": 4}, zorder=5)
    added_tp = round((n["recall"] - r["recall"]) * res["n_error"])
    added_fp = round((n["fpr"] - r["fpr"]) * res["n_correct"])
    ax.text(0.012, 1.11, f"Rules → fusion:\n+{added_tp} caught, +{added_fp} false alarms",
            fontsize=5.6, color=MUTED, ha="left", va="top", linespacing=1.2)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("Recall")
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_title("(a) Operating points", fontsize=7.5, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.27), ncol=2, fontsize=6, frameon=False,
              handlelength=2.2, columnspacing=0.8, labelspacing=0.3)

    # (b) per task, diverging: false alarms left, error detections right
    pt = res["per_task"]
    ys = np.arange(len(TASKS))[::-1]
    rules_c, gru_c = POINTS["Rules"][1], CURVES["GRU"][1]
    for yv, task in zip(ys, TASKS):
        t = pt[task]
        bx.barh(yv, t["tp_rules"], color=rules_c, height=0.62)
        bx.barh(yv, t["tp_gru_added"], left=t["tp_rules"], color=gru_c, height=0.62)
        bx.barh(yv, t["missed_by_fusion"], left=t["tp_rules"] + t["tp_gru_added"], color=MISSED, height=0.62)
        bx.barh(yv, -t["fp_rules"], color=rules_c, height=0.62, hatch="//////", edgecolor="white", linewidth=0)
        bx.barh(yv, -t["fp_gru_added"], left=-t["fp_rules"], color=gru_c, height=0.62,
                hatch="//////", edgecolor="white", linewidth=0)
    bx.axvline(0, color=INK, linewidth=0.7)
    n_err, n_ok = pt[TASKS[0]]["n_error"], pt[TASKS[0]]["n_correct"]
    bx.set_xlim(-14, n_err)
    ticks = [-10, -5, 0, 5, 10, 15, 20]
    bx.set_xticks(ticks)
    bx.set_xticklabels([str(abs(v)) for v in ticks])
    bx.set_yticks(ys)
    bx.set_yticklabels(TASKS)
    bx.set_xlabel("False alarms  |  Error episodes")
    bx.grid(axis="x", color=GRID, linewidth=0.5)
    bx.set_axisbelow(True)
    bx.set_title("(b) Detections per task", fontsize=7.5, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=rules_c), plt.Rectangle((0, 0), 1, 1, color=gru_c),
               plt.Rectangle((0, 0), 1, 1, color=MISSED)]
    bx.legend(handles, ["Rules", "Added by GRU", "Missed"], loc="upper center", bbox_to_anchor=(0.45, -0.27),
              ncol=3, fontsize=6, frameon=False, handlelength=1.2, columnspacing=1.0)
    for a in (ax, bx):
        for side in ["top", "right"]:
            a.spines[side].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(dest / "fig_tradeoff.pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(dest / "fig_tradeoff.png", dpi=220, bbox_inches="tight", pad_inches=0.02, metadata={"Software": None})
    plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_logging()
    res_dir, der, dest = Path(args.results), Path(args.derived), Path(args.dest)
    der.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(res_dir / "predictions.csv")
    res = compute(df)
    # Consistency with the confusion counts used by the tables
    conf = json.loads((der / "confusion.json").read_text(encoding="utf-8"))
    for meth in POINTS:
        c = conf[meth]
        assert abs(res["points"][meth]["recall"] - c["recall"]) < 1e-12
        assert abs(res["points"][meth]["fpr"] - c["false_positive_rate"]) < 1e-12
    assert sum(t["tp_rules"] for t in res["per_task"].values()) == conf["Rules"]["tp"]
    assert sum(t["tp_rules"] + t["tp_gru_added"] for t in res["per_task"].values()) == conf["NeSy"]["tp"]
    assert sum(t["fp_rules"] + t["fp_gru_added"] for t in res["per_task"].values()) == conf["NeSy"]["fp"]
    code_root = Path(__file__).resolve().parents[1]
    try:
        res["code_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root,
                                            capture_output=True, text=True).stdout.strip()
        res["code_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=code_root,
                                                capture_output=True, text=True).stdout.strip())
    except OSError:
        res["code_commit"], res["code_dirty"] = "unknown", True
    (der / "tradeoff.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    plot(res, dest)
    log.info("Trade-off figure written to %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
