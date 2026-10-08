"""
Export paper assets for the ADL error-detection study
=====================================================
Reads the outputs of run_adl_error_detection.py and writes:
  - macros.tex          : \\newcommand per reported number (no hand-typed numbers)
  - overall.tex, per_task_f1.tex : result tables
  - fig_scarcity.pdf    : F1 vs. number of error-labelled training participants

Usage (Bash):
  python -m scripts.export_adl_paper_assets \
      --results outputs/adl_errors --dest "../ICMSCT26/latex/generated"
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from adl_nesy.utils.logging import setup_logging

log = logging.getLogger(__name__)

# Categorical slots 1-5 of the validated reference palette (fixed order);
# markers and line styles give a second, colour-independent encoding.
SERIES = {
    "Rules": ("#2a78d6", "o", "-"),
    "IForest": ("#eb6834", "v", (0, (1, 1))),
    "RF": ("#1baf7a", "s", "--"),
    "GRU": ("#eda100", "^", "-."),
    "NeSy": ("#e87ba4", "D", (0, (5, 1, 1, 1, 1, 1))),
}
LABEL_FREE = ("Rules", "IForest")
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export LaTeX macros, tables and figures for the ADL error paper.")
    ap.add_argument("--results", type=str, default="outputs/adl_errors")
    ap.add_argument("--dest", type=str, required=True)
    return ap.parse_args()


def _macro_name(*parts: str) -> str:
    out = []
    for p in parts:
        p = p.replace("-", " ").replace("_", " ")
        out.append("".join(w[:1].upper() + w[1:] for w in p.split()))
    name = "".join(out)
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "".join(digits.get(c, c) for c in name)


def write_macros(
    summary: Dict[str, Any],
    per_task: pd.DataFrame,
    scar: pd.DataFrame | None,
    confound: Dict[str, Any],
    sens: Dict[str, Any] | None,
    dest: Path,
) -> None:
    m: Dict[str, str] = {}
    m["NEpisodes"] = str(summary["n_episodes"])
    m["NError"] = str(summary["n_error"])
    m["NCorrect"] = str(summary["n_correct"])
    m["NParticipants"] = str(summary["n_participants"])
    m["KFolds"] = str(summary["k_folds"])
    m["NSeeds"] = str(len(summary["seeds"]))
    for meth, s in summary["pooled"].items():
        for key in ["precision", "recall", "f1", "balanced_accuracy", "auroc"]:
            if key in s:
                m[_macro_name(meth, key)] = f"{s[key]:.2f}"
        m[_macro_name(meth, "false alarms")] = str(s["false_alarms"])
        m[_macro_name(meth, "missed")] = str(s["missed"])
        lo, hi = summary["f1_ci"][meth]
        m[_macro_name(meth, "f1 lo")] = f"{lo:.2f}"
        m[_macro_name(meth, "f1 hi")] = f"{hi:.2f}"
    for pair, d in summary["paired_f1_diff_ci"].items():
        a, b = pair.split("-", 1)
        # Math mode so negative values get a true minus sign
        m[_macro_name("diff", a, b)] = f"${d['diff']:+.2f}$"
        m[_macro_name("diff", a, b, "lo")] = f"${d['ci95'][0]:+.2f}$"
        m[_macro_name("diff", a, b, "hi")] = f"${d['ci95'][1]:+.2f}$"
    for meth, d in summary.get("seed_f1", {}).items():
        m[_macro_name("seed", meth, "mean")] = f"{d['mean']:.2f}"
        m[_macro_name("seed", meth, "std")] = f"{d['std']:.2f}"
    t = summary["trivial_all_error"]
    m["TrivialFOne"] = f"{t['f1']:.2f}"
    m["TrivialPrecision"] = f"{t['precision']:.2f}"
    e = summary["explanations"]
    m["NesyTrueDetections"] = str(e["nesy_true_detections"])
    m["NesyTrueWithReason"] = str(e["nesy_true_detections_with_rule_reason"])
    m["NesyReasonPct"] = f"{100 * e['nesy_true_detections_with_rule_reason'] / max(e['nesy_true_detections'], 1):.0f}"
    m["RuleTrueDetections"] = str(e["rule_true_detections"])
    m["RuleFaithful"] = str(e["rule_true_detections_matching_injected_error"])
    m["RuleFaithfulPct"] = f"{100 * e['rule_true_detections_matching_injected_error'] / max(e['rule_true_detections'], 1):.0f}"
    for k, v in summary["latency_ms_per_episode"].items():
        m[_macro_name("latency", k)] = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
    th = pd.DataFrame(summary["thresholds_per_fold"])
    for col in th.columns:
        m[_macro_name("thr", col, "min")] = f"{th[col].min():.2f}"
        m[_macro_name("thr", col, "max")] = f"{th[col].max():.2f}"
    for _, r in per_task.iterrows():
        m[_macro_name("task", r["task"], r["method"], "f1")] = f"{r['f1']:.2f}"
        m[_macro_name("task", r["task"], r["method"], "fa")] = str(int(r["false_alarms"]))
        m[_macro_name("task", r["task"], r["method"], "missed")] = str(int(r["missed"]))
    if scar is not None:
        g = scar.groupby(["error_participants", "method"])["f1"].mean()
        for (n, meth), v in g.items():
            m[_macro_name("scar", meth, str(n))] = f"{v:.2f}"
    fa_step = fa_post = 0
    for reason, n in e["rule_false_alarm_reasons"].items():
        if "Post_" in reason:
            fa_post += int(n)
        else:
            fa_step += int(n)
    m["RulesFaStep"] = str(fa_step)
    m["RulesFaPost"] = str(fa_post)
    m["RuleMismatched"] = str(e["rule_true_detections"] - e["rule_true_detections_matching_injected_error"])
    for sensor, d in confound.get("idle_level_first_reading", {}).items():
        m[_macro_name("idle p", sensor)] = f"{d['mannwhitney_p']:.2f}"
        m[_macro_name("idle err", sensor)] = f"{d['error_median']:.3f}"
        m[_macro_name("idle ok", sensor)] = f"{d['correct_median']:.3f}"

    if sens is not None:
        for q, s in sens["quantile"].items():
            tag = {"0.90": "Ninety", "0.95": "NinetyFive", "1.00": "Max"}[q]
            m[f"SensF{tag}"] = f"{s['f1']:.2f}"
            m[f"SensFa{tag}"] = str(s["false_alarms"])
        for k, s in sens["ablation"].items():
            tag = {"P1": "POne", "P2": "PTwo", "P1+P2": "Both"}[k]
            m[f"Abl{tag}FOne"] = f"{s['f1']:.2f}"
            m[f"Abl{tag}Recall"] = f"{s['recall']:.2f}"
            m[f"Abl{tag}Precision"] = f"{s['precision']:.2f}"

    lines = ["% Auto-generated by evaluation/export_adl_paper_assets.py -- do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (dest / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote %d macros", len(m))


DISPLAY = {
    "Rules": "Rules (ours)",
    "IForest": "Isolation forest, per task",
    "LogReg": "Logistic regression",
    "RF": "Random forest",
    "GRU": "GRU",
    "NeSy": "NeSy (Rules $\\lor$ GRU)",
    "GRU-motion": "GRU, motion events only",
}
TASKS = [
    ("PhoneCall", "PhoneCall (wrong number)"),
    ("WashHands", "WashHands (water left on)"),
    ("Cook", "Cook (burner left on)"),
    ("Eat", "Eat (no medicine)"),
    ("Clean", "Clean (no water)"),
]


def write_tables(summary: Dict[str, Any], per_task: pd.DataFrame, dest: Path) -> None:
    n_ok = summary["n_correct"]

    def row(name: str, s: Dict[str, Any], ci: Any) -> str:
        auroc = f"{s['auroc']:.2f}" if "auroc" in s else "--"
        f1 = f"{s['f1']:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
        return (f"{name} & {s['precision']:.2f} & {s['recall']:.2f} & {f1} & "
                f"{s['balanced_accuracy']:.2f} & {auroc} & {s['false_alarms']} \\\\")

    lines = [
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"Method & Prec. & Rec. & F1 [95\% CI] & Bal.\ acc. & AUROC & False alarms \\",
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{No error labels (correct executions only)}} \\",
    ]
    for m in ["Rules", "IForest"]:
        lines.append(row(DISPLAY[m], summary["pooled"][m], summary["f1_ci"][m]))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{@{}l}{\emph{Supervised with error labels}} \\")
    for m in ["LogReg", "RF", "GRU", "NeSy"]:
        lines.append(row(DISPLAY[m], summary["pooled"][m], summary["f1_ci"][m]))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{@{}l}{\emph{Controls}} \\")
    lines.append(row(DISPLAY["GRU-motion"], summary["pooled"]["GRU-motion"], summary["f1_ci"]["GRU-motion"]))
    lines.append(row("Flag every episode", summary["trivial_all_error"], None))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (dest / "overall.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cols = ["Rules", "IForest", "LogReg", "RF", "GRU", "NeSy"]
    lines = [
        r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{No error labels} & \multicolumn{4}{c}{Supervised} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-7}",
        r"Task (injected error) & Rules & IForest & LogReg & RF & GRU & NeSy \\",
        r"\midrule",
    ]
    for task, label in TASKS:
        sub = per_task[per_task["task"] == task].set_index("method")
        vals = [round(float(sub.at[c, "f1"]), 2) for c in cols]
        best = max(vals)
        cells = [f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}" for v in vals]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (dest / "per_task_f1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote tables (false alarms out of %d correct episodes)", n_ok)


def plot_scarcity(scar: pd.DataFrame, summary: Dict[str, Any], dest: Path) -> None:
    full_n = 16  # error-labelled participants in a training fold (20 - 4 held out)
    g = scar.groupby(["error_participants", "method"])["f1"].agg(["mean", "std"]).reset_index()
    full = {k: summary["pooled"][k]["f1"] for k in SERIES}

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    xs_all = sorted(g["error_participants"].unique()) + [full_n]
    for meth, (color, marker, ls) in SERIES.items():
        sub = g[g["method"] == meth].sort_values("error_participants")
        xs = list(sub["error_participants"]) + [full_n]
        ys = list(sub["mean"]) + [full[meth]]
        sd = list(sub["std"].fillna(0)) + [0.0]
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, linewidth=1.5, markersize=4.5, label=meth, zorder=3)
        if meth not in LABEL_FREE:
            ax.fill_between(xs, [y - s for y, s in zip(ys, sd)], [y + s for y, s in zip(ys, sd)], color=color, alpha=0.12, linewidth=0)
        ax.annotate(meth, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs_all)
    ax.set_xticklabels([str(x) if x != full_n else f"{full_n}\n(all)" for x in xs_all])
    ax.set_xlim(0.85, full_n * 1.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Error-labelled participants in training fold")
    ax.set_ylabel("F1 (error class)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncol=4, fontsize=7, handlelength=2.6)
    fig.tight_layout()
    fig.savefig(dest / "fig_scarcity.pdf")
    fig.savefig(dest / "fig_scarcity.png", dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_logging()
    res = Path(args.results)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    summary = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    confound = json.loads((res / "confound.json").read_text(encoding="utf-8"))
    per_task = pd.read_csv(res / "per_task.csv")
    scar = pd.read_csv(res / "scarcity.csv") if (res / "scarcity.csv").exists() else None

    sens_path = res / "rule_sensitivity.json"
    sens = json.loads(sens_path.read_text(encoding="utf-8")) if sens_path.exists() else None
    write_macros(summary, per_task, scar, confound, sens, dest)
    write_tables(summary, per_task, dest)
    if scar is not None:
        plot_scarcity(scar, summary, dest)
    log.info("Paper assets written to %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
